"""Regression test for parallel eval local state isolation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from hud.eval.context import EvalContext
from hud.eval.manager import run_eval


@pytest.mark.asyncio
async def test_parallel_eval_isolates_captured_locals_per_run() -> None:
    """Each parallel run should get an isolated copy of captured locals."""
    captured_state = {"counter": 0}

    with (
        patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
        patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        patch("hud.eval.manager._send_job_enter", new_callable=AsyncMock, return_value=None),
    ):
        async with run_eval(group=3, quiet=True) as ctx:
            # Accessing prompt raises ParallelEvalComplete on the summary context,
            # which skips re-running this block after parallel execution completes.
            _ = ctx.prompt

            captured_state["counter"] += 1
            await asyncio.sleep(0)
            ctx.reward = float(captured_state["counter"])

    rewards = [result.reward for result in ctx.results]
    assert rewards == [1.0, 1.0, 1.0]
