"""Behavioral tests for the local-path split incident.

Verifies:
1. registration guardrails still hold in ``hud.eval.manager``
2. local ``_run_evaluation(remote=False)`` forwards ``taskset`` into ``run_dataset()``
3. ``run_dataset(..., taskset=...)`` forwards taskset into ``hud.eval()``
4. ``JobEnterPayload`` serializes ``taskset`` into the outbound payload
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def run_in_new_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestRegistrationGuardrails:
    """Async registration must still be correct on the supporting path."""

    def test_send_job_enter_uses_async_http_client(self):
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"status": "ok", "task_version_ids": ["tv-1"]}

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("httpx.post", side_effect=AssertionError("sync httpx.post() should not be used")):
                with patch("httpx.AsyncClient", return_value=mock_client):
                    with patch("hud.settings.settings") as mock_settings:
                        mock_settings.telemetry_enabled = True
                        mock_settings.api_key = "test-key"
                        mock_settings.hud_api_url = "https://api.test"

                        result = await _send_job_enter(
                            job_id="job-1",
                            name="local eval",
                            variants=None,
                            group=1,
                            api_key="test-key",
                        )

            mock_client.post.assert_awaited_once()
            assert result == ["tv-1"]

        run_in_new_loop(_run())

    def test_missing_task_version_ids_does_not_crash_registration(self):
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"status": "ok"}

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("httpx.post", return_value=mock_resp):
                with patch("httpx.AsyncClient", return_value=mock_client):
                    with patch("hud.settings.settings") as mock_settings:
                        mock_settings.telemetry_enabled = True
                        mock_settings.api_key = "test-key"
                        mock_settings.hud_api_url = "https://api.test"

                        return await _send_job_enter(
                            job_id="job-2",
                            name="local eval",
                            variants=None,
                            group=1,
                            api_key="test-key",
                        )

        result = run_in_new_loop(_run())
        assert result is None, (
            "When the registration response omits task_version_ids, the async registration "
            "path should return None instead of crashing."
        )


class TestLocalEntryPathDispatch:
    """The local CLI path must forward taskset into run_dataset()."""

    def test_local_run_evaluation_passes_taskset_to_run_dataset(self):
        from hud.cli.eval import EvalConfig, _run_evaluation

        dummy_task = SimpleNamespace(id="task-1")
        run_dataset_mock = AsyncMock(return_value=[SimpleNamespace(reward=1.0)])

        async def _run():
            cfg = EvalConfig(source="tasks.json", agent_type="claude", taskset="nightly-suite")

            with patch("hud.datasets.load_tasks", return_value=[dummy_task]):
                with patch("hud.datasets.run_dataset", run_dataset_mock):
                    with patch.object(EvalConfig, "get_agent_kwargs", return_value={}):
                        await _run_evaluation(cfg)

        run_in_new_loop(_run())

        kwargs = run_dataset_mock.await_args.kwargs
        assert kwargs.get("taskset") == "nightly-suite", (
            "Local _run_evaluation(remote=False) must forward cfg.taskset into run_dataset()."
        )


class TestRunnerAndPayloadPropagation:
    """The local taskset must survive runner dispatch and payload serialization."""

    def test_run_dataset_forwards_taskset_to_hud_eval(self):
        from hud.datasets.runner import run_dataset

        captured: dict[str, object] = {}
        dummy_task = SimpleNamespace(id="task-1")
        ctx = SimpleNamespace(system_prompt=None, results=[])
        fake_agent = SimpleNamespace(run=AsyncMock(return_value=None))
        fake_agent_type = SimpleNamespace(cls=SimpleNamespace(create=MagicMock(return_value=fake_agent)))

        @asynccontextmanager
        async def fake_eval(tasks, **kwargs):
            captured["tasks"] = tasks
            captured["kwargs"] = kwargs
            yield ctx

        async def _run():
            with patch("hud.datasets.loader.load_tasks", return_value=[dummy_task]):
                with patch("hud.datasets.runner.hud.eval", fake_eval):
                    await run_dataset(
                        "tasks.json",
                        fake_agent_type,
                        max_steps=3,
                        group_size=2,
                        quiet=True,
                        taskset="nightly-suite",
                    )

        run_in_new_loop(_run())

        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs.get("taskset") == "nightly-suite", (
            "run_dataset(..., taskset=...) must forward taskset into hud.eval()."
        )

    def test_job_enter_payload_serializes_taskset(self):
        from hud.eval.types import JobEnterPayload

        payload = JobEnterPayload(name="local eval", taskset="nightly-suite")
        data = payload.model_dump(exclude_none=True)

        assert data.get("taskset") == "nightly-suite", (
            "JobEnterPayload.model_dump(exclude_none=True) must include the taskset field "
            "when taskset is provided."
        )
