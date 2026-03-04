from __future__ import annotations

import asyncio

import pytest

from hud.eval.context import set_trace_context
from hud.native.comparator import CompareTool, ComparisonMode
from hud.tools.submit import SubmitTool, set_submission


@pytest.fixture(autouse=True)
def reset_submission_state() -> None:
    """Keep submission storage clean between tests."""
    set_submission(None)
    with set_trace_context("trace-a"):
        set_submission(None)
    with set_trace_context("trace-b"):
        set_submission(None)
    yield
    set_submission(None)
    with set_trace_context("trace-a"):
        set_submission(None)
    with set_trace_context("trace-b"):
        set_submission(None)


@pytest.mark.asyncio
async def test_submission_fallback_is_isolated_per_trace() -> None:
    """Concurrent fallback comparisons should not leak across traces."""
    submit = SubmitTool()
    compare = CompareTool()

    a_submitted = asyncio.Event()
    allow_a_compare = asyncio.Event()

    async def worker_a():
        with set_trace_context("trace-a"):
            await submit(response="worker-a-answer")
            a_submitted.set()
            await allow_a_compare.wait()
            return await compare(
                value=None,
                reference="worker-a-answer",
                mode=ComparisonMode.EXACT,
            )

    async def worker_b():
        await a_submitted.wait()
        with set_trace_context("trace-b"):
            await submit(response="worker-b-answer")
            b_result = await compare(
                value=None,
                reference="worker-b-answer",
                mode=ComparisonMode.EXACT,
            )
        allow_a_compare.set()
        return b_result

    a_result, b_result = await asyncio.gather(worker_a(), worker_b())

    assert b_result.done
    assert b_result.reward == 1.0
    assert a_result.done
    assert a_result.reward == 1.0


@pytest.mark.asyncio
async def test_explicit_value_path_remains_stable() -> None:
    """Control test: explicit values should stay deterministic."""
    compare = CompareTool()
    result = await compare(
        value="stable-answer",
        reference="stable-answer",
        mode=ComparisonMode.EXACT,
    )

    assert result.done
    assert result.reward == 1.0
