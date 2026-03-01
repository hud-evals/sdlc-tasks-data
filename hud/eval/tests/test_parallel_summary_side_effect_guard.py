"""Regression test for summary re-execution side effects in parallel eval."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from hud.eval.context import EvalContext
from hud.eval.manager import run_eval


@pytest.mark.asyncio
async def test_parallel_eval_summary_pass_does_not_reexecute_user_side_effects() -> None:
    """Summary context must not execute user body side effects an extra time."""
    side_effects = {"count": 0}

    with (
        patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
        patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        patch("hud.eval.manager._send_job_enter", new_callable=AsyncMock, return_value=None),
    ):
        async with run_eval(group=3, quiet=True) as ctx:
            # Accessing an allowlisted attribute on summary context currently does not
            # raise ParallelEvalComplete, so this body can run one extra time.
            _ = ctx.reward

            side_effects["count"] += 1
            await asyncio.sleep(0)
            ctx.reward = float(side_effects["count"])

    assert len(ctx.results) == 3
    assert side_effects["count"] == 3
