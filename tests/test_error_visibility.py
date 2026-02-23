"""Behavioral integration tests for remote eval error visibility.

These tests validate end-to-end behavior through the remote execution flow:
- `_run_evaluation` should fail when every rollout is rejected.
- `_run_evaluation` should succeed when at least one rollout is accepted.
- `eval_command` should translate submission failures into a non-zero CLI exit.

The tests intentionally avoid constraining implementation shape (list/dict/model).
They only assert externally visible behavior.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


def _remote_cfg():
    """Create a minimal remote EvalConfig for integration tests."""
    from hud.cli.eval import EvalConfig
    from hud.types import AgentType

    return EvalConfig(
        source="dummy.json",
        agent_type=AgentType.CLAUDE,
        remote=True,
        all=True,
        max_steps=10,
        group_size=1,
    )


def _patch_remote_dependencies(tasks: list[dict], responses: list[dict]):
    """Patch external dependencies used by remote evaluation path."""
    mock_send_job_enter = AsyncMock(return_value=None)
    mock_client = _make_mock_client(responses)

    @contextmanager
    def _ctx():
        with (
            patch("hud.datasets.load_tasks", return_value=tasks),
            patch("hud.eval.manager._send_job_enter", new=mock_send_job_enter),
            patch("hud.datasets.utils.httpx.AsyncClient", return_value=mock_client),
            patch("hud.cli.eval.settings") as eval_settings,
            patch("hud.datasets.utils.settings") as ds_settings,
        ):
            yield eval_settings, ds_settings

    return _ctx()


class TestRemoteRunBehavior:
    """Behavior-driven integration tests for remote eval flow."""

    def test_all_rejected_surfaces_failure(self):
        """Remote run must fail when every submitted task is rejected."""
        from hud.cli.eval import _run_evaluation

        tasks = [
            {"id": "t1", "env": {"name": "browser"}, "scenario": "s1"},
            {"id": "t2", "env": {"name": "browser"}, "scenario": "s2"},
        ]
        responses = [{
            "accepted": 0,
            "rejected": 2,
            "results": [
                {"status": "rejected", "error": "bad task 1"},
                {"status": "rejected", "error": "bad task 2"},
            ],
        }]

        with _patch_remote_dependencies(tasks, responses) as (eval_settings, ds_settings):
            eval_settings.api_key = "test-key"
            ds_settings.api_key = "test-key"
            ds_settings.hud_api_url = "https://api.test"

            with pytest.raises(ValueError):
                _run(_run_evaluation(_remote_cfg()))

    def test_partial_acceptance_is_successful(self):
        """Remote run should succeed if at least one task is accepted."""
        from hud.cli.eval import _run_evaluation

        tasks = [
            {"id": "t1", "env": {"name": "browser"}, "scenario": "s1"},
            {"id": "t2", "env": {"name": "browser"}, "scenario": "s2"},
        ]
        responses = [{
            "accepted": 1,
            "rejected": 1,
            "results": [
                {"trace_id": "tr-1", "status": "accepted"},
                {"status": "rejected", "error": "bad task 2"},
            ],
        }]

        with _patch_remote_dependencies(tasks, responses) as (eval_settings, ds_settings):
            eval_settings.api_key = "test-key"
            ds_settings.api_key = "test-key"
            ds_settings.hud_api_url = "https://api.test"

            results, loaded_tasks = _run(_run_evaluation(_remote_cfg()))
            assert results == []
            assert len(loaded_tasks) == 2

    def test_all_accepted_is_successful(self):
        """Remote run should succeed when all submitted tasks are accepted."""
        from hud.cli.eval import _run_evaluation

        tasks = [{"id": "t1", "env": {"name": "browser"}, "scenario": "s1"}]
        responses = [{
            "accepted": 1,
            "rejected": 0,
            "results": [{"trace_id": "tr-1", "status": "accepted"}],
        }]

        with _patch_remote_dependencies(tasks, responses) as (eval_settings, ds_settings):
            eval_settings.api_key = "test-key"
            ds_settings.api_key = "test-key"
            ds_settings.hud_api_url = "https://api.test"

            results, loaded_tasks = _run(_run_evaluation(_remote_cfg()))
            assert results == []
            assert len(loaded_tasks) == 1


class TestCliFailureMapping:
    """CLI should expose remote submission failure via non-zero exit code."""

    def test_eval_command_exits_nonzero_on_value_error(self):
        """ValueError from evaluation path must translate to typer.Exit(1)."""
        from hud.cli.eval import eval_command

        with (
            patch("hud.cli.eval._run_evaluation", new=AsyncMock(side_effect=ValueError("submit failed"))),
            patch("hud.cli.eval.settings") as eval_settings,
        ):
            eval_settings.api_key = "test-key"

            with pytest.raises(Exception) as exc_info:
                eval_command(
                    source="dummy.json",
                    agent="claude",
                    all=True,
                    remote=True,
                    yes=True,
                )

            assert getattr(exc_info.value, "exit_code", None) == 1
