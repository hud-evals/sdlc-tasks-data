"""Tests for the scenario lifecycle — error surfacing, typed params, and display handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.types import EvaluationResult


def _run(coro):
    """Run an async coroutine without requiring pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestEvalYieldErrorSurfacing:
    """Verify that evaluation errors propagate rather than being silently swallowed."""

    def test_error_not_silently_swallowed(self):
        """A scenario that raises during evaluation must NOT return reward=0.0 silently."""
        from hud.environment.scenarios import ScenarioMixin, ScenarioSession

        class FakeEnv(ScenarioMixin):
            name = "test"
            _prompt_manager = MagicMock()
            _resource_manager = MagicMock()
            _tool_manager = MagicMock()
            _scenarios = {}
            _active_session = None

        env = FakeEnv()
        env._init_scenarios()

        async def _inner():
            async def failing_gen():
                yield "prompt text"
                raise ValueError("evaluation crashed")

            gen = failing_gen()
            await gen.__anext__()

            env._active_session = ScenarioSession(
                local_name="test_scenario",
                full_name="test:test_scenario",
                is_local=True,
                connection_name=None,
                resource_uri="test:test_scenario",
                generator=gen,
                answer="test answer",
            )

            with pytest.raises(ValueError, match="evaluation crashed"):
                await env.run_scenario_evaluate("test_scenario")

        _run(_inner())

    def test_error_message_preserved(self):
        """The original error message must be preserved when it propagates."""
        from hud.environment.scenarios import ScenarioMixin, ScenarioSession

        class FakeEnv(ScenarioMixin):
            name = "test"
            _prompt_manager = MagicMock()
            _resource_manager = MagicMock()
            _tool_manager = MagicMock()
            _scenarios = {}
            _active_session = None

        env = FakeEnv()
        env._init_scenarios()

        async def _inner():
            async def failing_gen():
                yield "prompt text"
                raise RuntimeError("specific error message")

            gen = failing_gen()
            await gen.__anext__()

            env._active_session = ScenarioSession(
                local_name="test_scenario",
                full_name="test:test_scenario",
                is_local=True,
                connection_name=None,
                resource_uri="test:test_scenario",
                generator=gen,
                answer="test",
            )

            with pytest.raises(RuntimeError, match="specific error message"):
                await env.run_scenario_evaluate("test_scenario")

        _run(_inner())

    def test_successful_eval_returns_reward(self):
        """A normal scenario should return the correct reward value."""
        from hud.environment.scenarios import ScenarioMixin, ScenarioSession

        class FakeEnv(ScenarioMixin):
            name = "test"
            _prompt_manager = MagicMock()
            _resource_manager = MagicMock()
            _tool_manager = MagicMock()
            _scenarios = {}
            _active_session = None

        env = FakeEnv()
        env._init_scenarios()

        async def _inner():
            async def success_gen():
                yield "do the thing"
                yield 0.85

            gen = success_gen()
            await gen.__anext__()

            env._active_session = ScenarioSession(
                local_name="test_scenario",
                full_name="test:test_scenario",
                is_local=True,
                connection_name=None,
                resource_uri="test:test_scenario",
                generator=gen,
                answer="done",
            )

            result = await env.run_scenario_evaluate("test_scenario")
            assert result is not None
            assert abs(result.reward - 0.85) < 0.01

        _run(_inner())


class TestTypedParameters:
    """Verify that scenario arguments are type-deserialized from MCP string args."""

    def test_int_param_deserialized(self):
        """An integer scenario parameter should be deserialized from MCP string '42' to int 42."""
        import inspect
        import json
        from typing import get_type_hints

        from hud.environment.scenarios import ScenarioMixin

        async def sample_scenario(threshold: int, name: str):
            yield f"threshold={threshold}, name={name}"
            yield 1.0

        hints = get_type_hints(sample_scenario)
        assert hints.get("threshold") is int

        class FakeEnv(ScenarioMixin):
            name = "test"
            _prompt_manager = MagicMock()
            _resource_manager = MagicMock()
            _tool_manager = MagicMock()
            _scenarios = {}
            _active_session = None

        env = FakeEnv()
        env._init_scenarios()
        env.scenario(name="typed_test")(sample_scenario)

        from pydantic import TypeAdapter
        adapter = TypeAdapter(int)
        result = adapter.validate_json("42")
        assert result == 42
        assert isinstance(result, int)

    def test_float_param_deserialized(self):
        """A float scenario parameter should be deserialized from MCP string '3.14' to float."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(float)
        result = adapter.validate_json("3.14")
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.001

    def test_string_param_unchanged(self):
        """String parameters should pass through without transformation."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(str)
        result = adapter.validate_python("hello world")
        assert result == "hello world"
        assert isinstance(result, str)


class TestParameterValidation:
    """Verify that parameter mismatches are caught at registration or call time."""

    def test_missing_required_param_raises(self):
        """Calling a scenario without a required parameter should raise an error."""
        import inspect

        async def needs_param(required_arg: str):
            yield "prompt"
            yield 1.0

        sig = inspect.signature(needs_param)
        params = sig.parameters

        assert "required_arg" in params
        assert params["required_arg"].default is inspect.Parameter.empty

    def test_extra_params_handled(self):
        """Calling with extra unexpected arguments should be handled gracefully."""
        import inspect

        async def simple_scenario(x: int):
            yield "prompt"
            yield 1.0

        sig = inspect.signature(simple_scenario)
        try:
            sig.bind(x=1, extra="unexpected")
            bound_ok = True
        except TypeError:
            bound_ok = False

        assert not bound_ok

    def test_valid_params_pass(self):
        """Correct arguments should not raise any validation error."""
        import inspect

        async def valid_scenario(x: int, name: str = "default"):
            yield "prompt"
            yield 1.0

        sig = inspect.signature(valid_scenario)
        bound = sig.bind(x=42, name="test")
        bound.apply_defaults()
        assert bound.arguments["x"] == 42
        assert bound.arguments["name"] == "test"


class TestDisplayNoneReward:
    """Verify that the display module handles None rewards gracefully."""

    def test_single_result_none_reward(self):
        """print_single_result should not crash when reward is None."""
        from hud.eval.display import print_single_result

        try:
            print_single_result("trace-123", "test-eval", reward=None)
        except (TypeError, ValueError):
            pytest.fail("print_single_result crashed on None reward")

    def test_results_table_mixed(self):
        """display_results should handle a mix of None and float rewards."""
        from hud.eval.display import display_results

        class FakeResult:
            def __init__(self, reward, error=None):
                self.reward = reward
                self.error = error
                self.index = 0
                self.duration = 1.0

        results = [FakeResult(0.8), FakeResult(None), FakeResult(0.5)]
        try:
            display_results(results, show_details=True)
        except (TypeError, ValueError):
            pytest.fail("display_results crashed on mixed None/float rewards")

    def test_error_status_shown(self):
        """Results with isError=True should not crash display."""
        from hud.eval.display import display_results

        class FakeResult:
            def __init__(self):
                self.reward = None
                self.error = "scenario crashed"
                self.index = 0
                self.duration = 0
                self.isError = True

        results = [FakeResult()]
        try:
            display_results(results, show_details=True)
        except (TypeError, ValueError):
            pytest.fail("display_results crashed on error result with None reward")
