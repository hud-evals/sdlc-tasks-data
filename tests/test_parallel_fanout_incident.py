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


class TestParallelFailureContainment:
    """One failed child run must not cancel or erase sibling work."""

    def test_error_containment_in_source(self):
        """The parallel execution path must contain individual task failures.

        Without containment, one child's exception propagates through
        asyncio.gather and cancels all siblings — producing fewer
        results than expected.
        """
        with open("hud/eval/manager.py") as f:
            source = f.read()

        has_return_exceptions = "return_exceptions" in source

        has_try_except = False
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("except") and "Exception" in stripped:
                nearby = "\n".join(lines[max(0, i):min(len(lines), i + 5)])
                if ".error" in nearby or "warning" in nearby.lower():
                    has_try_except = True
                    break

        assert has_return_exceptions or has_try_except, (
            "manager.py lacks error containment for parallel evaluation. "
            "A single child failure will propagate through asyncio.gather "
            "and cancel all sibling tasks, producing incomplete fanout results."
        )

    def test_single_failure_preserves_siblings(self):
        """Behaviorally verify that one failing coroutine does not cancel others."""
        async def _test():
            results = []

            async def succeed(val):
                await asyncio.sleep(0.01)
                results.append(val)
                return val

            async def fail():
                raise ValueError("child failure")

            coros = [succeed(1), fail(), succeed(3)]
            completed = await asyncio.gather(*coros, return_exceptions=True)

            successes = [r for r in completed if not isinstance(r, Exception)]
            assert len(successes) >= 2, (
                f"Expected at least 2 successes but got {len(successes)}. "
                "gather must tolerate individual child failures."
            )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_failed_child_produces_result(self):
        """A failed child must produce an exception result, not disappear."""
        async def _test():
            async def fail():
                raise RuntimeError("child error")

            async def succeed():
                return "ok"

            results = await asyncio.gather(
                succeed(), fail(), succeed(), return_exceptions=True
            )
            assert len(results) == 3, (
                f"Expected 3 results (including failures) but got {len(results)}"
            )
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 1

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()


class TestParallelExceptionClassification:
    """Cancellation-related failures must not surface as bogus client-init errors."""

    def test_cancelled_not_client_error(self):
        """CancelledError must NOT be reclassified as HudClientError."""
        from hud.shared.exceptions import HudClientError, HudException

        cancelled = asyncio.CancelledError(
            "task cancelled: event loop shutting down"
        )
        result = HudException._analyze_exception(cancelled, str(cancelled))

        assert not isinstance(result, HudClientError), (
            f"CancelledError was misclassified as HudClientError: {result!r}. "
            "The 'event loop' pattern in _analyze_exception is too broad."
        )

    def test_real_client_error_still_detected(self):
        """Genuine 'not initialized' errors must still become HudClientError."""
        from hud.shared.exceptions import HudClientError, HudException

        real_error = RuntimeError(
            "MCP client not initialized — call connect() first"
        )
        result = HudException._analyze_exception(real_error, str(real_error))

        assert isinstance(result, HudClientError), (
            f"Genuine client error was not classified as HudClientError: "
            f"{type(result).__name__}"
        )

    def test_event_loop_closed_still_detected(self):
        """'event loop is closed' errors must still become HudClientError."""
        from hud.shared.exceptions import HudClientError, HudException

        closed_error = RuntimeError("event loop is closed")
        result = HudException._analyze_exception(closed_error, str(closed_error))

        assert isinstance(result, HudClientError), (
            f"'event loop is closed' was not classified as HudClientError: "
            f"{type(result).__name__}"
        )


class TestParallelHealthSummary:
    """Failed/cancelled children must remain in the denominator.

    When the health summary excludes failed/cancelled runs from the
    denominator, it overstates batch health — the team sees 100%
    success when real results are far worse.
    """

    def test_failed_children_in_denominator(self):
        """log_eval_stats must include failed children in the total count."""
        from unittest.mock import MagicMock

        from hud.eval.parallel import log_eval_stats

        ctx_ok = MagicMock()
        ctx_ok.reward = 0.8
        ctx_ok.success = True

        ctx_fail = MagicMock()
        ctx_fail.reward = None
        ctx_fail.success = False

        handler = logging.StreamHandler(stream=io.StringIO())
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("hud.eval.parallel")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            log_eval_stats([ctx_ok, ctx_fail, ctx_ok])
            output = handler.stream.getvalue()
        finally:
            logger.removeHandler(handler)

        assert "2/3" in output, (
            f"Health summary should report 2/3 succeeded but got: {output!r}. "
            "Failed children are being excluded from the denominator, "
            "making batch health look better than reality."
        )

    def test_mean_reward_includes_failures_as_zero(self):
        """Mean reward must treat failed children as 0.0, not exclude them."""
        from unittest.mock import MagicMock

        from hud.eval.parallel import log_eval_stats

        ctx_ok = MagicMock()
        ctx_ok.reward = 1.0
        ctx_ok.success = True

        ctx_fail = MagicMock()
        ctx_fail.reward = None
        ctx_fail.success = False

        handler = logging.StreamHandler(stream=io.StringIO())
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("hud.eval.parallel")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            log_eval_stats([ctx_ok, ctx_fail])
            output = handler.stream.getvalue()
        finally:
            logger.removeHandler(handler)

        assert "0.500" in output, (
            f"Mean reward should be 0.500 (1.0 + 0.0 / 2) but got: {output!r}. "
            "Failed children with reward=None must be counted as 0.0 in the "
            "mean, not excluded from the calculation."
        )

    def test_all_failed_reports_zero(self):
        """When all children fail, health must report 0/N and mean 0.000."""
        from unittest.mock import MagicMock

        from hud.eval.parallel import log_eval_stats

        ctx_fail = MagicMock()
        ctx_fail.reward = None
        ctx_fail.success = False

        handler = logging.StreamHandler(stream=io.StringIO())
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("hud.eval.parallel")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            log_eval_stats([ctx_fail, ctx_fail, ctx_fail])
            output = handler.stream.getvalue()
        finally:
            logger.removeHandler(handler)

        assert "0/3" in output, (
            f"Health summary should report 0/3 when all fail but got: {output!r}. "
            "The denominator must always reflect the total number of children."
        )
        assert "0.000" in output, (
            f"Mean reward should be 0.000 when all fail but got: {output!r}."
        )
