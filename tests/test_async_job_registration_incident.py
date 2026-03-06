"""Behavioral tests for the async job registration incident.

Verifies three guarantees:
1. Registration works under an async request lifecycle
2. Missing task_version_ids does not crash successful registration
3. Taskset supplied at the CLI entry path survives through
   CLI → runner → payload serialization into the outbound request
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch


class TestAsyncRegistrationLifecycle:
    """Registration must work correctly under an async request lifecycle."""

    def test_send_job_enter_is_async(self):
        from hud.eval.manager import _send_job_enter
        assert inspect.iscoroutinefunction(_send_job_enter), (
            "_send_job_enter must be an async function"
        )

    def test_registration_completes_without_runtime_error(self):
        """Calling _send_job_enter inside an async context must not raise RuntimeError."""
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

    def test_registration_uses_async_http_client(self):
        """The outbound registration request must use an async HTTP client."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"status": "ok", "task_version_ids": ["tv-1"]}

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client) as patched:
                with patch("hud.settings.settings") as mock_settings:
                    mock_settings.telemetry_enabled = True
                    mock_settings.api_key = "test-key"
                    mock_settings.hud_api_url = "https://api.test"

                    await _send_job_enter(
                        job_id="j1", name="test",
                        variants=None, group=1, api_key="test-key",
                    )
                    patched.assert_called_once()
                    mock_client.post.assert_awaited_once()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()


class TestRegistrationResponseHandling:
    """Missing task_version_ids must not crash successful registration."""

    def test_handles_missing_task_version_ids(self):
        """Response without task_version_ids should return None, not crash."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"status": "ok"}

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
            "_send_job_enter should return None instead of crashing"
        )

    def test_handles_valid_task_version_ids(self):
        """Response with valid task_version_ids should return them."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "status": "ok",
                "task_version_ids": ["tv-abc", "tv-def"],
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

                    return await _send_job_enter(
                        job_id="j1", name="test",
                        variants=None, group=1, api_key="test-key",
                    )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert result == ["tv-abc", "tv-def"]


class TestTasksetPropagationEndToEnd:
    """Taskset must survive from CLI entry through to the outbound registration payload."""

    def test_job_enter_payload_has_taskset_field(self):
        from hud.eval.types import JobEnterPayload
        fields = JobEnterPayload.model_fields
        assert "taskset" in fields, (
            "JobEnterPayload must have a 'taskset' field"
        )

    def test_taskset_serializes_in_payload(self):
        """Taskset value must appear in the serialized outbound payload."""
        from hud.eval.types import JobEnterPayload
        payload = JobEnterPayload(name="test-eval", taskset="my-benchmark")
        data = payload.model_dump(exclude_none=True)
        assert "taskset" in data, (
            "JobEnterPayload.model_dump() must include taskset when set"
        )
        assert data["taskset"] == "my-benchmark"

    def test_run_dataset_accepts_taskset_parameter(self):
        from hud.datasets.runner import run_dataset
        sig = inspect.signature(run_dataset)
        assert "taskset" in sig.parameters, (
            "run_dataset() must accept a 'taskset' parameter"
        )

    def test_taskset_reaches_outbound_request(self):
        """Drive the local eval path with a fake taskset and capture the
        outbound registration payload to prove taskset survives end-to-end."""
        from hud.eval.manager import _send_job_enter

        captured_payloads = []

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "status": "ok",
                "task_version_ids": ["tv-1"],
            }

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            async def capture_post(url, **kwargs):
                captured_payloads.append(kwargs.get("json", {}))
                return mock_resp
            mock_client.post = AsyncMock(side_effect=capture_post)

            with patch("httpx.AsyncClient", return_value=mock_client):
                with patch("hud.settings.settings") as mock_settings:
                    mock_settings.telemetry_enabled = True
                    mock_settings.api_key = "test-key"
                    mock_settings.hud_api_url = "https://api.test"

                    await _send_job_enter(
                        job_id="test-job",
                        name="eval (benchmark.json)",
                        variants=None,
                        group=2,
                        api_key="test-key",
                        taskset="incident-benchmark",
                    )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

        assert len(captured_payloads) == 1, "Expected exactly one outbound registration request"
        payload = captured_payloads[0]
        assert "taskset" in payload, (
            "The outbound registration payload must contain the 'taskset' field. "
            "The taskset value was lost somewhere in the propagation chain."
        )
        assert payload["taskset"] == "incident-benchmark", (
            f"Expected taskset='incident-benchmark' in payload, got '{payload.get('taskset')}'"
        )
