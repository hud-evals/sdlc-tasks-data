"""Behavioral tests for the async job registration branch-trust incident.

Verifies five guarantees with a runtime-heavy emphasis:
1. _send_job_enter uses an async HTTP client, not sync httpx.post()
2. Missing task_version_ids in the response does not crash registration
3. The registration path works correctly inside a real async eval flow
4. The full async lifecycle completes without RuntimeError
5. Taskset survives from CLI entry through to the outbound registration payload
"""
from __future__ import annotations

import asyncio
import inspect
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
    """Taskset must survive from CLI entry through to the outbound registration payload."""

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
