"""Behavioral tests for the async job registration branch-trust incident.

Verifies five guarantees with a runtime-heavy emphasis:
1. _send_job_enter uses an async HTTP client, not sync httpx.post()
2. Missing task_version_ids in the response does not crash registration
3. The registration path works correctly inside a real async eval flow
4. The local CLI path forwards taskset into run_dataset()
5. The runner/hud.eval boundary and payload schema preserve taskset
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class TestRegistrationRuntimeLifecycle:
    """Registration must use an async HTTP transport and handle responses safely."""

    def test_async_client_path_is_used(self):
        """Sync httpx.post must not be called; the async client path must succeed."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "status": "ok",
                "job_id": "test-job-1",
                "task_version_ids": ["tv-1", "tv-2"],
            }

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            def sync_post_trap(*args, **kwargs):
                raise AssertionError(
                    "sync httpx.post() was called inside _send_job_enter — "
                    "this causes RuntimeError in async contexts"
                )

            with patch("httpx.post", side_effect=sync_post_trap):
                with patch("httpx.AsyncClient", return_value=mock_client):
                    with patch("hud.settings.settings") as mock_settings:
                        mock_settings.telemetry_enabled = True
                        mock_settings.api_key = "test-key"
                        mock_settings.hud_api_url = "https://api.test"

                        result = await _send_job_enter(
                            job_id="j1", name="test-eval",
                            variants=None, group=1, api_key="test-key",
                        )
                        return result

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert result is not None
        assert isinstance(result, list)

    def test_missing_task_version_ids_does_not_crash(self):
        """A 200 OK response without task_version_ids must return None, not raise KeyError."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"status": "ok", "job_id": "abc123"}

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client):
                with patch("hud.settings.settings") as mock_settings:
                    mock_settings.telemetry_enabled = True
                    mock_settings.api_key = "test-key"
                    mock_settings.hud_api_url = "https://api.test"

                    result = await _send_job_enter(
                        job_id="j1", name="test",
                        variants=None, group=1, api_key="test-key",
                    )
                    return result

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert result is None, (
            "When task_version_ids is absent from the response, "
            "_send_job_enter should return None instead of raising KeyError"
        )

    def test_registration_completes_without_runtime_error(self):
        """Calling _send_job_enter inside an active async event loop must succeed."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "status": "ok",
                "task_version_ids": ["tv-1", "tv-2"],
            }

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client):
                with patch("hud.settings.settings") as mock_settings:
                    mock_settings.telemetry_enabled = True
                    mock_settings.api_key = "test-key"
                    mock_settings.hud_api_url = "https://api.test"

                    result = await _send_job_enter(
                        job_id="j1", name="test-eval",
                        variants=None, group=1, api_key="test-key",
                    )
                    return result

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2


class TestRegistrationRuntimeEndToEnd:
    """The real local registration flow must not fail with async/runtime symptoms."""

    def test_local_registration_flow_under_async_context(self):
        """Drive the local registration path under an active event loop with mocked HTTP.

        This rejects agents who copied the teammate branch propagation fixes
        but never validated the actual Sentry-visible runtime half.
        """
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "status": "ok",
                "job_id": "integration-test-job",
                "task_version_ids": ["tv-int-1"],
            }

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            def sync_post_trap(*args, **kwargs):
                raise RuntimeError(
                    "This event loop is already running — sync httpx.post() "
                    "was used inside the async registration path"
                )

            with patch("httpx.post", side_effect=sync_post_trap):
                with patch("httpx.AsyncClient", return_value=mock_client) as patched_ac:
                    with patch("hud.settings.settings") as mock_settings:
                        mock_settings.telemetry_enabled = True
                        mock_settings.api_key = "test-key"
                        mock_settings.hud_api_url = "https://api.test"

                        result = await _send_job_enter(
                            job_id="integration-j1",
                            name="eval (integration-test.json)",
                            variants={"model": "test-model"},
                            group=2,
                            api_key="test-key",
                            taskset="integration-suite",
                        )

                        patched_ac.assert_called_once()
                        mock_client.post.assert_awaited_once()

                        return result

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert result == ["tv-int-1"]


class TestTasksetPropagationEndToEnd:
    """Taskset must survive across the real local forwarding chain."""

    def test_cli_and_runner_preserve_taskset(self):
        """Drive the real local entry path in two hops:

        1. `_run_evaluation()` must pass `cfg.taskset` into `run_dataset()`
        2. `run_dataset()` must pass `taskset` into `hud.eval(...)`, and the
           payload schema must preserve the field for registration serialization
        """
        from hud.cli.eval import EvalConfig, _run_evaluation
        from hud.datasets.runner import run_dataset
        from hud.eval.task import Task
        from hud.eval.types import JobEnterPayload

        async def _run():
            fake_task = Task(env={"name": "test"}, id="task-1", scenario="test")

            mock_run_dataset = AsyncMock(return_value=[])
            with (
                patch("hud.datasets.load_tasks", return_value=[fake_task]),
                patch("hud.datasets.run_dataset", mock_run_dataset),
            ):
                cfg = EvalConfig(
                    source="dummy.json",
                    agent_type="integration_test",
                    all=True,
                    quiet=True,
                    taskset="incident-benchmark",
                )
                await _run_evaluation(cfg)

            mock_run_dataset.assert_awaited_once()
            assert mock_run_dataset.await_args.kwargs["taskset"] == "incident-benchmark", (
                "The local CLI path must forward cfg.taskset into run_dataset()."
            )

            captured_eval_kwargs = {}
            fake_ctx = SimpleNamespace(system_prompt=None, results=[], reward=None)

            @asynccontextmanager
            async def fake_eval(*args, **kwargs):
                captured_eval_kwargs.update(kwargs)
                yield fake_ctx

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value=None)
            mock_agent_cls = MagicMock()
            mock_agent_cls.create.return_value = mock_agent

            with (
                patch("hud.datasets.runner.hud.eval", new=fake_eval),
                patch("hud.agents.misc.integration_test_agent.IntegrationTestRunner", mock_agent_cls),
            ):
                await run_dataset(
                    [fake_task],
                    agent_type="integration_test",
                    quiet=True,
                    taskset="incident-benchmark",
                )

            assert captured_eval_kwargs["taskset"] == "incident-benchmark", (
                "run_dataset() must forward taskset into hud.eval(...)."
            )

            payload = JobEnterPayload(
                name="eval (benchmark.json)",
                group=2,
                taskset="incident-benchmark",
            )
            dumped = payload.model_dump(exclude_none=True)
            assert dumped["taskset"] == "incident-benchmark", (
                "JobEnterPayload must preserve taskset for registration serialization."
            )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()
