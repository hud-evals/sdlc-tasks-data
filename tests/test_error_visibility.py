"""Integration tests for remote eval error visibility.

Verifies that submit_rollouts returns useful submission information
(not None/fire-and-forget) and that _send_job_enter propagates errors.

Tests are implementation-agnostic: they accept any return type (list, dict,
Pydantic model, dataclass, etc.) as long as the behavioral contract is met.
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    """Run an async coroutine without requiring pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_client(responses: list[dict]):
    """Create a mock httpx.AsyncClient that returns the given JSON responses."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_responses = []
    for resp_data in responses:
        resp = MagicMock()
        resp.json.return_value = resp_data
        resp.raise_for_status = MagicMock()
        mock_responses.append(resp)

    mock_client.post = AsyncMock(side_effect=mock_responses)
    return mock_client


def _get_accepted_count(result) -> int | None:
    """Extract accepted task count from any valid return format.

    Supports: list (len), object with total_accepted/accepted attr,
    dict with total_accepted/accepted key.  Returns None only if
    the result is None itself (fire-and-forget).
    """
    if result is None:
        return None

    if isinstance(result, list):
        return len(result)

    for attr in ("total_accepted", "accepted"):
        val = getattr(result, attr, None)
        if isinstance(val, int):
            return val

    if isinstance(result, dict):
        for key in ("total_accepted", "accepted"):
            if key in result and isinstance(result[key], int):
                return result[key]

    return 0


class TestSubmitRolloutsReturnsInfo:
    """submit_rollouts must return useful submission info, not None."""

    def test_not_fire_and_forget(self):
        """submit_rollouts must return something (not None) when tasks are accepted."""
        from hud.datasets.utils import submit_rollouts
        from hud.types import AgentType

        async def _test():
            mock_client = _make_mock_client([{
                "accepted": 1, "rejected": 0,
                "results": [{"trace_id": "trace-abc", "status": "accepted"}],
            }])

            with (
                patch("hud.datasets.utils.httpx.AsyncClient", return_value=mock_client),
                patch("hud.settings.settings") as mock_settings,
            ):
                mock_settings.telemetry_enabled = True
                mock_settings.api_key = "test-key"
                mock_settings.hud_api_url = "https://api.test"

                result = await submit_rollouts(
                    tasks=[{"env": {"name": "browser"}, "scenario": "test"}],
                    agent_type=AgentType.CLAUDE,
                    job_id="job-123",
                )

            assert result is not None, (
                "submit_rollouts returned None — still fire-and-forget. "
                "It must return submission results so callers know what happened."
            )

        _run(_test())

    def test_indicates_acceptance(self):
        """Return value must indicate at least 1 task was accepted."""
        from hud.datasets.utils import submit_rollouts
        from hud.types import AgentType

        async def _test():
            mock_client = _make_mock_client([{
                "accepted": 1, "rejected": 0,
                "results": [{"trace_id": "trace-abc", "status": "accepted"}],
            }])

            with (
                patch("hud.datasets.utils.httpx.AsyncClient", return_value=mock_client),
                patch("hud.settings.settings") as mock_settings,
            ):
                mock_settings.telemetry_enabled = True
                mock_settings.api_key = "test-key"
                mock_settings.hud_api_url = "https://api.test"

                result = await submit_rollouts(
                    tasks=[{"env": {"name": "browser"}, "scenario": "test"}],
                    agent_type=AgentType.CLAUDE,
                    job_id="job-123",
                )

            accepted = _get_accepted_count(result)
            assert accepted is not None, (
                "submit_rollouts returned None — still fire-and-forget."
            )
            assert accepted >= 1, (
                f"Expected at least 1 accepted task, got {accepted}. "
                f"Return value: {result!r}"
            )

        _run(_test())

    def test_multiple_accepted_tasks(self):
        """Multiple accepted tasks must all be reflected in the result."""
        from hud.datasets.utils import submit_rollouts
        from hud.types import AgentType

        async def _test():
            mock_client = _make_mock_client([{
                "accepted": 3, "rejected": 0,
                "results": [
                    {"trace_id": "t-1", "status": "accepted"},
                    {"trace_id": "t-2", "status": "accepted"},
                    {"trace_id": "t-3", "status": "accepted"},
                ],
            }])

            with (
                patch("hud.datasets.utils.httpx.AsyncClient", return_value=mock_client),
                patch("hud.settings.settings") as mock_settings,
            ):
                mock_settings.telemetry_enabled = True
                mock_settings.api_key = "test-key"
                mock_settings.hud_api_url = "https://api.test"

                result = await submit_rollouts(
                    tasks=[
                        {"env": {"name": "browser"}, "scenario": f"test-{i}"}
                        for i in range(3)
                    ],
                    agent_type=AgentType.CLAUDE,
                    job_id="job-456",
                )

            accepted = _get_accepted_count(result)
            assert accepted is not None, (
                "submit_rollouts returned None — still fire-and-forget."
            )
            assert accepted >= 3, (
                f"Submitted 3 tasks, all accepted, but result shows {accepted}. "
                f"Return value: {result!r}"
            )

        _run(_test())

    def test_return_type_annotation_changed(self):
        """submit_rollouts must not have -> None return annotation."""
        from hud.datasets.utils import submit_rollouts

        sig = inspect.signature(submit_rollouts)
        ret = sig.return_annotation

        assert ret is not None, (
            "submit_rollouts has no return annotation."
        )
        assert ret is not inspect.Parameter.empty, (
            "submit_rollouts has no return annotation."
        )
        assert ret is not type(None), (
            "submit_rollouts still annotated as -> None (fire-and-forget)."
        )
        ret_str = str(ret) if not isinstance(ret, str) else ret
        assert ret_str.lower() != "none", (
            "submit_rollouts still annotated as -> None (fire-and-forget)."
        )


class TestSubmitRolloutsHandlesRejection:
    """When all tasks are rejected, submit_rollouts must surface the failure."""

    def test_all_rejected_raises_or_returns_zero(self):
        """All-rejected must raise an exception or return with zero accepted."""
        from hud.datasets.utils import submit_rollouts
        from hud.types import AgentType

        async def _test():
            mock_client = _make_mock_client([{
                "accepted": 0, "rejected": 2,
                "results": [
                    {"status": "rejected", "error": "bad task 1"},
                    {"status": "rejected", "error": "bad task 2"},
                ],
            }])

            with (
                patch("hud.datasets.utils.httpx.AsyncClient", return_value=mock_client),
                patch("hud.settings.settings") as mock_settings,
            ):
                mock_settings.telemetry_enabled = True
                mock_settings.api_key = "test-key"
                mock_settings.hud_api_url = "https://api.test"

                try:
                    result = await submit_rollouts(
                        tasks=[
                            {"env": {"name": "browser"}, "scenario": "test-1"},
                            {"env": {"name": "browser"}, "scenario": "test-2"},
                        ],
                        agent_type=AgentType.CLAUDE,
                        job_id="job-000",
                    )
                except (RuntimeError, ValueError, SystemExit, Exception):
                    return  # raising is valid

                accepted = _get_accepted_count(result)
                if accepted is None:
                    raise AssertionError(
                        "All tasks rejected but submit_rollouts returned None. "
                        "Must raise or return a result indicating zero accepted."
                    )
                assert accepted == 0, (
                    f"All tasks rejected but result indicates {accepted} accepted. "
                    f"Return value: {result!r}"
                )

        _run(_test())


class TestJobEnterStillWorks:
    """Verify _send_job_enter (already fixed by teammate) still raises on failure."""

    def test_raises_on_http_error(self):
        """_send_job_enter must raise when the API returns an error."""
        from hud.eval.manager import _send_job_enter
        import httpx

        async def _test():
            with (
                patch("httpx.AsyncClient") as mock_client_cls,
                patch("hud.settings.settings") as mock_settings,
            ):
                mock_settings.telemetry_enabled = True
                mock_settings.api_key = "test"
                mock_settings.hud_api_url = "https://api.test"

                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)

                error_response = MagicMock(spec=httpx.Response)
                error_response.status_code = 500
                error_response.is_success = False
                error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Server Error",
                    request=MagicMock(),
                    response=error_response,
                )
                mock_client.post.return_value = error_response
                mock_client_cls.return_value = mock_client

                await _send_job_enter(
                    job_id="job-1",
                    name="test",
                    variants=None,
                    group=1,
                    api_key="test",
                )

        raised = False
        try:
            _run(_test())
        except Exception:
            raised = True
        assert raised, (
            "_send_job_enter did not raise on HTTP 500. "
            "It must propagate errors instead of silently returning None."
        )
