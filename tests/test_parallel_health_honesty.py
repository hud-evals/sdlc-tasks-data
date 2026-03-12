"""Behavioral tests for the grouped health-honesty incident variant.

This suite keeps health-summary truthfulness as the dominant signal while still
requiring the containment and classifier guardrails that make the summary
trustworthy in production.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
from contextlib import contextmanager

from hud.eval.context import EvalContext
from hud.eval.parallel import log_eval_stats
from hud.shared.exceptions import HudClientError, HudException


def _run(coro):
    """Run an async helper without depending on pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ctx(*, reward: float | None, error: BaseException | None = None) -> EvalContext:
    """Create a real EvalContext carrying the fields used by grouped summaries."""
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


def _logged_fraction(output: str) -> tuple[int, int]:
    match = re.search(r"(\d+)/(\d+) succeeded", output)
    assert match, f"Could not parse success fraction from log output: {output!r}"
    return int(match.group(1)), int(match.group(2))


def _logged_mean_reward(output: str) -> float:
    match = re.search(r"mean_reward=([0-9]+\.[0-9]+)", output)
    assert match, f"Could not parse mean_reward from log output: {output!r}"
    return float(match.group(1))


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


class TestGroupedHealthDenominatorIntegrity:
    """Failed and cancelled children must still shape grouped-health math."""

    def test_mixed_outcomes_keep_all_children_in_denominator(self):
        completed = [
            _ctx(reward=1.0),
            _ctx(reward=0.5),
            _ctx(reward=0.25),
            _ctx(reward=None, error=RuntimeError("boom")),
            _ctx(reward=None, error=asyncio.CancelledError("cancelled")),
        ]

        with _capture_parallel_log() as stream:
            log_eval_stats(completed)
        output = stream.getvalue()

        assert _logged_fraction(output) == (3, 5), (
            "Grouped health should keep all settled children in the denominator."
        )
        assert abs(_logged_mean_reward(output) - 0.35) < 0.001, (
            "Failed/cancelled children should contribute 0.0 only at summary time."
        )

    def test_all_failed_or_cancelled_reports_zero_of_total_not_zero_of_zero(self):
        completed = [
            _ctx(reward=None, error=RuntimeError("boom-1")),
            _ctx(reward=None, error=RuntimeError("boom-2")),
            _ctx(reward=None, error=asyncio.CancelledError("cancelled")),
        ]

        with _capture_parallel_log() as stream:
            log_eval_stats(completed)
        output = stream.getvalue()

        assert _logged_fraction(output) == (0, 3), (
            "All-failed grouped summaries must still report the real batch size."
        )
        assert abs(_logged_mean_reward(output) - 0.0) < 0.001

    def test_failed_children_contribute_zero_without_mutating_raw_contexts(self):
        failed = _ctx(reward=None, error=RuntimeError("boom"))
        cancelled = _ctx(reward=None, error=asyncio.CancelledError("cancelled"))
        succeeded = _ctx(reward=1.0)
        completed = [failed, cancelled, succeeded]

        with _capture_parallel_log():
            log_eval_stats(completed)

        assert failed.reward is None, (
            "Summary code must not rewrite failed child contexts to reward=0.0."
        )
        assert cancelled.reward is None, (
            "Cancelled child contexts must remain truthful at the raw-result layer."
        )


class TestGroupedHealthLogsMatchRawOutcomes:
    """The published rollup must reconcile with the actual child outcomes."""

    def test_logged_success_denominator_matches_completed_contexts(self):
        completed = [
            _ctx(reward=0.9),
            _ctx(reward=0.6),
            _ctx(reward=0.4),
            _ctx(reward=None, error=RuntimeError("boom")),
            _ctx(reward=None, error=asyncio.CancelledError("cancelled")),
        ]

        with _capture_parallel_log() as stream:
            log_eval_stats(completed, context="grouped rollout")
        output = stream.getvalue()

        assert "(grouped rollout)" in output
        assert _logged_fraction(output) == (3, 5), (
            "Logged success counts should reconcile with the raw grouped outcomes."
        )

    def test_logged_mean_reward_matches_raw_outcomes(self):
        completed = [
            _ctx(reward=1.0),
            _ctx(reward=0.8),
            _ctx(reward=0.2),
            _ctx(reward=None, error=RuntimeError("boom")),
            _ctx(reward=None, error=asyncio.CancelledError("cancelled")),
        ]

        with _capture_parallel_log() as stream:
            log_eval_stats(completed)
        output = stream.getvalue()

        expected_mean = (1.0 + 0.8 + 0.2 + 0.0 + 0.0) / 5
        assert abs(_logged_mean_reward(output) - expected_mean) < 0.001, (
            "Logged mean reward must match raw outcomes, including zero-valued failures."
        )


class TestContainmentAndClassificationGuardrails:
    """Supporting fixes are required so grouped summaries stay trustworthy."""

    def test_single_failure_preserves_siblings_on_real_manager_path(self, monkeypatch):
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
            assert by_index[1].error is not None, "The failed child must remain visibly failed."
            assert by_index[1].reward is None, "Containment should not forge a synthetic success-like reward."

        _run(_test())

    def test_cancelled_not_client_error_but_event_loop_closed_still_is(self):
        cancelled = asyncio.CancelledError("task cancelled: event loop shutting down")
        cancelled_result = _classify(cancelled)
        closed_result = _classify(RuntimeError("event loop is closed"))

        assert not isinstance(cancelled_result, HudClientError), (
            "Cancellation-derived noise should not surface as a client-init failure."
        )
        assert isinstance(closed_result, HudClientError), (
            "Legitimate 'event loop is closed' signals must remain detectable."
        )

    def test_real_client_error_still_detected(self):
        result = _classify(RuntimeError("MCP client not initialized - call connect() first"))

        assert isinstance(result, HudClientError), (
            "True client initialization failures should still classify as HudClientError."
        )
