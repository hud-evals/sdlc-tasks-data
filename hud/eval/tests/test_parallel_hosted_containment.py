from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from hud.environment import Environment
from hud.eval.context import EvalContext
from hud.eval.manager import run_eval
from hud.eval.task import Task


def _make_hosted_task() -> Task:
    return Task(env={"name": "browser"}, args={})


def _make_local_task() -> Task:
    return Task(env=Environment("local-only"), args={})


@pytest.mark.asyncio
async def test_grouped_hud_hosted_runs_fail_closed_at_run_eval_entrypoint() -> None:
    task = _make_hosted_task()

    with patch("hud.eval.manager._send_job_enter", new_callable=AsyncMock, return_value=None):
        with pytest.raises(
            RuntimeError,
            match="Grouped runs against HUD-hosted remote environments are temporarily blocked",
        ) as excinfo:
            async with run_eval([task], group=2, quiet=True) as ctx:
                assert ctx is not None
                pytest.fail("Grouped HUD-hosted runs should fail closed before execution starts")

    assert "cross-trace contamination risk" in str(excinfo.value)
    assert "group=1" in str(excinfo.value)


@pytest.mark.asyncio
async def test_grouped_hud_hosted_variant_expansion_fails_closed() -> None:
    task = _make_hosted_task()

    with patch("hud.eval.manager._send_job_enter", new_callable=AsyncMock, return_value=None):
        with pytest.raises(
            RuntimeError,
            match="Grouped runs against HUD-hosted remote environments are temporarily blocked",
        ):
            async with run_eval(task, variants={"model": ["gpt-4o", "claude"]}, quiet=True) as ctx:
                assert ctx is not None
                pytest.fail("Hosted variant expansion should fail closed before execution starts")

@pytest.mark.asyncio
async def test_grouped_hud_hosted_group_fails_closed() -> None:
    task = _make_hosted_task()

    with patch("hud.eval.manager._send_job_enter", new_callable=AsyncMock, return_value=None):
        with pytest.raises(
            RuntimeError,
            match="Grouped runs against HUD-hosted remote environments are temporarily blocked",
        ):
            async with run_eval(task, group=3, quiet=True) as ctx:
                assert ctx is not None
                pytest.fail("Hosted grouped execution should fail closed before execution starts")


@pytest.mark.asyncio
async def test_single_trace_hud_hosted_run_remains_allowed() -> None:
    task = _make_hosted_task()

    with (
        patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
        patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        patch.object(EvalContext, "_run_task_scenario_setup", new_callable=AsyncMock),
        patch.object(EvalContext, "__aenter__", new_callable=AsyncMock) as mock_enter,
        patch.object(EvalContext, "__aexit__", new_callable=AsyncMock) as mock_exit,
    ):
        mock_enter.return_value = EvalContext(name="test")
        mock_exit.return_value = False
        async with run_eval(task, quiet=True) as ctx:
            assert ctx is not None


@pytest.mark.asyncio
async def test_grouped_local_runs_are_not_blocked() -> None:
    task = _make_local_task()

    with (
        patch("hud.eval.manager._send_job_enter", new_callable=AsyncMock, return_value=None),
        patch("hud.eval.manager._run_parallel_eval", new_callable=AsyncMock) as mock_parallel,
    ):
        mock_parallel.return_value = []
        try:
            async with run_eval([task], group=3, quiet=True) as ctx:
                assert ctx is not None
        except Exception:
            pass
        mock_parallel.assert_called_once()
