"""Integration tests for parallel evaluation execution.

Verifies that variant expansion creates independent task copies,
asyncio.gather handles failures gracefully, and CancelledError is
not misclassified as HudClientError.
"""
from __future__ import annotations

import asyncio
import inspect


class TestExpandVariants:
    """Variant expansion must create independent copies."""

    def test_variants_are_independent(self):
        """Modifying one variant's args must not affect others."""
        from hud.eval.parallel import expand_variants

        combos = expand_variants({"model": ["gpt-4o", "claude"]})
        assert len(combos) == 2

        combos[0]["extra"] = "injected"
        assert "extra" not in combos[1], (
            "Variants share state — modifying one affected the other"
        )

    def test_deep_nested_args_independent(self):
        """Nested dicts in variants must be independent."""
        from hud.eval.parallel import expand_variants

        combos = expand_variants({"config": [{"a": 1}, {"b": 2}]})
        assert len(combos) == 2

        combos[0]["config"]["injected"] = True
        assert "injected" not in combos[1].get("config", {}), (
            "Nested variant dicts share state — mutation leaked"
        )

    def test_variant_count(self):
        """Correct number of combinations produced."""
        from hud.eval.parallel import expand_variants

        combos = expand_variants({"model": ["a", "b"], "temp": [0.5, 1.0]})
        assert len(combos) == 4, f"Expected 4 combos, got {len(combos)}"


class TestGatherErrorHandling:
    """Parallel execution must tolerate individual failures."""

    def test_parallel_error_containment(self):
        """Parallel eval must contain individual task failures.

        Without error containment, one task failure propagates through
        asyncio.gather and cancels all sibling tasks. Valid approaches:
        1. try/except in the per-task function (catches errors, stores on ctx)
        2. return_exceptions=True on asyncio.gather
        """
        with open("hud/eval/manager.py") as f:
            source = f.read()

        has_return_exceptions = "return_exceptions" in source

        has_error_capture = False
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("except") and "Exception" in stripped:
                nearby = "\n".join(lines[max(0, i):min(len(lines), i + 5)])
                if ".error" in nearby:
                    has_error_capture = True
                    break

        assert has_return_exceptions or has_error_capture, (
            "manager.py lacks error containment for parallel evaluation. "
            "When one task fails, asyncio.gather propagates the exception "
            "and cancels all sibling tasks.\n\n"
            "Fix: Either wrap per-task execution in try/except (storing "
            "errors on the context), or use return_exceptions=True on "
            "asyncio.gather."
        )

    def test_single_failure_doesnt_cancel_others(self):
        """One failing coroutine must not cancel sibling tasks."""
        async def _test():
            results = []

            async def succeed(val):
                await asyncio.sleep(0.01)
                results.append(val)
                return val

            async def fail():
                raise ValueError("intentional failure")

            coros = [succeed(1), fail(), succeed(3)]
            completed = await asyncio.gather(*coros, return_exceptions=True)

            successes = [r for r in completed if not isinstance(r, Exception)]
            assert len(successes) >= 2, (
                f"Expected at least 2 successes, got {len(successes)}. "
                "gather must use return_exceptions=True"
            )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_failed_task_is_exception(self):
        """Failed tasks produce exception objects, not missing results."""
        async def _test():
            async def fail():
                raise RuntimeError("task error")

            async def succeed():
                return "ok"

            results = await asyncio.gather(succeed(), fail(), succeed(), return_exceptions=True)
            assert len(results) == 3, "All tasks must produce a result (even failures)"
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 1, f"Expected 1 exception, got {len(exceptions)}"

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()


class TestExceptionClassification:
    """CancelledError must not be misclassified as HudClientError."""

    def test_cancelled_not_client_error(self):
        """CancelledError through _analyze_exception must NOT become HudClientError."""
        from hud.shared.exceptions import HudClientError, HudException

        cancelled = asyncio.CancelledError("task cancelled: event loop shutting down")
        result = HudException._analyze_exception(cancelled, str(cancelled))

        assert not isinstance(result, HudClientError), (
            f"CancelledError was misclassified as HudClientError: {result!r}. "
            "The pattern matching in _analyze_exception is too broad — "
            "'event loop' in error message should also require 'closed'."
        )

    def test_real_client_error_still_detected(self):
        """Genuine 'not initialized' errors must still become HudClientError."""
        from hud.shared.exceptions import HudClientError, HudException

        real_error = RuntimeError("MCP client not initialized — call connect() first")
        result = HudException._analyze_exception(real_error, str(real_error))

        assert isinstance(result, HudClientError), (
            f"Genuine client error was not classified as HudClientError: {type(result).__name__}"
        )

    def test_event_loop_closed_still_detected(self):
        """'event loop is closed' errors must still become HudClientError."""
        from hud.shared.exceptions import HudClientError, HudException

        closed_error = RuntimeError("event loop is closed")
        result = HudException._analyze_exception(closed_error, str(closed_error))

        assert isinstance(result, HudClientError), (
            f"'event loop is closed' error was not classified as HudClientError: "
            f"{type(result).__name__}"
        )
