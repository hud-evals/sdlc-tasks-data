"""Behavioral tests for the parallel fanout incident.

Verifies three independent failure families:
1. Failure containment — one child failure must not poison siblings
2. Exception classification — cancellation must not masquerade as client-init
3. Health summary — failed/cancelled children must stay in the denominator
"""
from __future__ import annotations

import asyncio
import io
import logging
from contextlib import contextmanager

from hud.eval.context import EvalContext
from hud.eval.parallel import log_eval_stats
from hud.shared.exceptions import HudClientError, HudException


def _run(coro):
    """Run an async test helper without requiring pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ctx(*, reward: float | None, error: BaseException | None = None) -> EvalContext:
    """Create a real EvalContext with the outcome fields the summary uses."""
    ctx = EvalContext(name="eval", trace=False, quiet=True)
    ctx.reward = reward
    ctx.error = error
    return ctx


def _classify(exc: BaseException):
    return HudException._analyze_exception(exc, str(exc))


@contextmanager
def _capture_parallel_log():
    logger = logging.getLogger("hud.eval.parallel")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


async def _run_parallel_case(monkeypatch, body, *, group: int = 3, max_concurrent: int | None = None):
    import importlib

    manager = importlib.import_module("hud.eval.manager")

    monkeypatch.setattr(manager, "find_user_frame", lambda: object())
    monkeypatch.setattr(
        manager,
        "get_with_block_body",
        lambda _frame: ("await __parallel_body__(ctx)", {"__parallel_body__": body}, "ctx"),
    )
    monkeypatch.setattr(manager, "print_eval_stats", lambda *args, **kwargs: None)

    return await manager._run_parallel_eval(
        tasks=[],
        variant_combos=[{}],
        group=group,
        group_ids=None,
        job_id="job-123",
        api_key=None,
        code_snippet="hidden test",
        max_concurrent=max_concurrent,
        trace=False,
        quiet=True,
    )


class TestParallelFailureContainment:
    """One failed child run must not cancel or erase sibling work."""

    def test_single_failure_preserves_siblings_on_real_manager_path(self, monkeypatch):
        """Run the actual fanout helper and prove siblings still finish."""

        async def _test():
            started: set[int] = set()
            finished: list[int] = []
            all_started = asyncio.Event()

            async def body(ctx):
                started.add(ctx.index)
                if len(started) == 3:
                    all_started.set()

                await asyncio.wait_for(all_started.wait(), timeout=1)

                if ctx.index == 1:
                    raise RuntimeError("child failure")

                await asyncio.sleep(0.01)
                ctx.reward = {0: 1.0, 2: 0.5}[ctx.index]
                finished.append(ctx.index)

            results = await _run_parallel_case(monkeypatch, body, group=3)
            by_index = {ctx.index: ctx for ctx in results}

            assert sorted(by_index) == [0, 1, 2], "All child runs should still be represented."
            assert sorted(finished) == [0, 2], "Healthy siblings should still finish their work."
            assert by_index[0].error is None
            assert by_index[2].error is None
            assert by_index[1].error is not None, "The failing child should still be surfaced as a failed result."
            assert by_index[0].reward == 1.0
            assert by_index[1].reward is None
            assert by_index[2].reward == 0.5

        _run(_test())


class TestParallelExceptionClassification:
    """Cancellation-related failures must not surface as bogus client-init errors."""

    def test_cancelled_not_client_error(self):
        """CancelledError must NOT be reclassified as HudClientError."""
        cancelled = asyncio.CancelledError("task cancelled: event loop shutting down")
        result = _classify(cancelled)

        assert not isinstance(result, HudClientError), (
            f"CancelledError was misclassified as HudClientError: {result!r}. "
            "Cancellation-derived noise should not masquerade as client initialization failures."
        )

    def test_real_client_error_still_detected(self):
        """Genuine 'not initialized' errors must still become HudClientError."""
        result = _classify(RuntimeError("MCP client not initialized - call connect() first"))

        assert isinstance(result, HudClientError), (
            f"Genuine client error was not classified as HudClientError: {type(result).__name__}"
        )

    def test_event_loop_closed_still_detected(self):
        """'event loop is closed' errors must still become HudClientError."""
        result = _classify(RuntimeError("event loop is closed"))

        assert isinstance(result, HudClientError), (
            f"'event loop is closed' was not classified as HudClientError: {type(result).__name__}"
        )


class TestParallelHealthSummary:
    """Failed/cancelled children must remain in the denominator."""

    def test_mixed_outcomes_report_truthful_health(self):
        """Mixed outcomes should report both the real success rate and mean reward."""
        completed = [
            _ctx(reward=1.0),
            _ctx(reward=None, error=RuntimeError("boom")),
            _ctx(reward=0.5),
        ]

        with _capture_parallel_log() as stream:
            log_eval_stats(completed)
        output = stream.getvalue()

        assert "2/3 succeeded" in output, (
            f"Health summary should report 2/3 succeeded but got: {output!r}."
        )
        assert "mean_reward=0.500" in output, (
            f"Mean reward should be 0.500 ((1.0 + 0.0 + 0.5) / 3) but got: {output!r}."
        )

    def test_all_failed_reports_zero_of_total(self):
        """When all children fail, health must still report the full batch size."""
        completed = [
            _ctx(reward=None, error=RuntimeError("boom-1")),
            _ctx(reward=None, error=RuntimeError("boom-2")),
            _ctx(reward=None, error=RuntimeError("boom-3")),
        ]

        with _capture_parallel_log() as stream:
            log_eval_stats(completed)
        output = stream.getvalue()

        assert "0/3 succeeded" in output, (
            f"Health summary should report 0/3 when all fail but got: {output!r}."
        )
        assert "mean_reward=0.000" in output, (
            f"Mean reward should be 0.000 when all fail but got: {output!r}."
        )
