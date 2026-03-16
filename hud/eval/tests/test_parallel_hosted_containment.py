from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from hud.environment import Environment
from hud.eval.manager import run_eval
from hud.eval.task import Task


class _FakeEvalContext:
    def __init__(self) -> None:
        self.eval_name = "eval"
        self.reward = None
        self.error = None

    async def __aenter__(self) -> _FakeEvalContext:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


def _make_hud_hosted_task() -> Task:
    env = Environment("browser")
    env._hub_config = {"name": "browser"}
    return Task(env=env)


def _make_non_hud_remote_task() -> Task:
    env = Environment("external")
    env.connect_url("https://mcp.example.com", alias="example")
    return Task(env=env)


def _make_local_task() -> Task:
    env = Environment("local")
    env.connect_mcp_config(
        {
            "filesystem": {
                "command": "python",
                "args": ["-m", "example_server"],
            }
        }
    )
    return Task(env=env)


@pytest.mark.asyncio
async def test_grouped_hud_hosted_runs_fail_closed() -> None:
    task = _make_hud_hosted_task()

    with (
        patch("hud.eval.manager._send_job_enter", new=AsyncMock(return_value=None)),
        patch("hud.eval.manager._run_parallel_eval", new=AsyncMock(return_value=[])),
        pytest.raises(
            RuntimeError,
            match="Grouped runs against HUD-hosted remote environments are temporarily blocked",
        ) as excinfo,
    ):
        async with run_eval(task, group=2, trace=False, quiet=True):
            pass

    assert "cross-trace contamination risk" in str(excinfo.value)
    assert "group=1" in str(excinfo.value)


@pytest.mark.asyncio
async def test_single_trace_hud_hosted_runs_remain_allowed() -> None:
    task = _make_hud_hosted_task()
    fake_ctx = _FakeEvalContext()

    with patch("hud.eval.context.EvalContext.from_task", return_value=fake_ctx):
        async with run_eval(task, group=1, trace=False, quiet=True) as ctx:
            assert ctx is fake_ctx


@pytest.mark.asyncio
async def test_grouped_non_hud_remote_runs_are_not_blocked() -> None:
    task = _make_non_hud_remote_task()

    with (
        patch("hud.eval.manager._send_job_enter", new=AsyncMock(return_value=None)),
        patch("hud.eval.manager._run_parallel_eval", new=AsyncMock(return_value=[])) as mock_run,
    ):
        async with run_eval(task, group=2, trace=False, quiet=True) as ctx:
            assert ctx.eval_name == "eval"

    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_grouped_local_runs_are_not_blocked() -> None:
    task = _make_local_task()

    with (
        patch("hud.eval.manager._send_job_enter", new=AsyncMock(return_value=None)),
        patch("hud.eval.manager._run_parallel_eval", new=AsyncMock(return_value=[])) as mock_run,
    ):
        async with run_eval(task, group=3, trace=False, quiet=True) as ctx:
            assert ctx.eval_name == "eval"

    mock_run.assert_awaited_once()
