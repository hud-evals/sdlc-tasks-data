"""Behavioral tests for the classifier-precision incident variant."""
from __future__ import annotations

import asyncio
import io
import logging
from contextlib import contextmanager
from pathlib import Path

from hud.eval.context import EvalContext
from hud.eval.parallel import log_eval_stats
from hud.shared.exceptions import HudClientError, HudException


def _run(coro):
    """Run async helpers without requiring pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ctx(*, reward: float | None, error: BaseException | None = None) -> EvalContext:
    ctx = EvalContext(name="eval", trace=False, quiet=True)
    ctx.reward = reward
    ctx.error = error
    return ctx


def _classify(exc: BaseException):
    return HudException._analyze_exception(exc, str(exc))


def _exceptions_source() -> str:
    return Path("hud/shared/exceptions.py").read_text()


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


async def _run_parallel_case(monkeypatch, body, *, group: int = 3):
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
        max_concurrent=None,
        trace=False,
        quiet=True,
    )


class TestExceptionClassificationPrecision:
    """Grouped shutdown noise must stop looking like startup/client failures."""

    def test_cancelled_error_stops_short_of_client_error(self):
        cancelled = asyncio.CancelledError("task cancelled: event loop shutting down")
        result = _classify(cancelled)

        assert not isinstance(result, HudClientError), (
            f"CancelledError was misclassified as HudClientError: {result!r}."
        )

    def test_event_loop_guard_is_narrowed_in_source(self):
        source = _exceptions_source()

        assert '"event loop" in error_msg and "closed" in error_msg' in source, (
            "The event-loop classifier should explicitly preserve the closed-loop path "
            "instead of matching every event-loop string."
        )

    def test_event_loop_closed_still_detected(self):
        result = _classify(RuntimeError("event loop is closed"))

        assert isinstance(result, HudClientError), (
            f"'event loop is closed' was not classified as HudClientError: {type(result).__name__}"
        )

    def test_real_client_init_error_still_detected(self):
        result = _classify(RuntimeError("MCP client not initialized - call connect() first"))

        assert isinstance(result, HudClientError), (
            f"Genuine client-init failure was not classified as HudClientError: {type(result).__name__}"
        )


class TestIncidentCorroborators:
    """Containment and health still need to be fixed end-to-end."""

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
            assert by_index[0].error is None
            assert by_index[2].error is None
            assert by_index[1].error is not None, "The failed child should remain visible as failed."
            assert by_index[0].reward == 1.0
            assert by_index[1].reward is None
            assert by_index[2].reward == 0.5

        _run(_test())

    def test_mixed_outcomes_report_truthful_health(self):
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
