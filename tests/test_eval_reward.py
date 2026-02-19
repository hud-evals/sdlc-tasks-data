"""Integration tests for the eval reward pipeline.

These tests verify BEHAVIOR — that rewards flow correctly from tool
results through the eval context to the final score, and that CLI
flags compose correctly. They accept any correct solution regardless
of the specific code pattern used.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _read(relpath: str) -> str:
    """Read a source file relative to the repo root."""
    return open(relpath).read()


# ---------------------------------------------------------------------------
# 1. structuredContent must flow through _execute_tool
# ---------------------------------------------------------------------------


class TestExecuteToolStructuredContent:
    """_execute_tool must forward structuredContent from tool results.

    We call the real _execute_tool method with mocked internals and verify
    that the returned MCPToolResult has structuredContent set.
    """

    def _build_env(self, *, is_local: bool, mock_result):
        """Build an Environment instance with mocked internals."""
        from hud.environment.environment import Environment

        env = object.__new__(Environment)
        env._mock_mode = False
        env._tool_routing_built = True

        env._router = MagicMock()
        env._router.is_local.return_value = is_local

        if is_local:
            env._tool_manager = AsyncMock()
            env._tool_manager.call_tool.return_value = mock_result
            env._router.get_connection.return_value = None
        else:
            env._router.get_connection.return_value = "test_conn"
            mock_conn = AsyncMock()
            mock_conn.call_tool.return_value = mock_result
            env._connections = {"test_conn": mock_conn}

        return env

    def test_local_tool_preserves_structured_content(self):
        """Local tool results must include structuredContent in the MCPToolResult."""
        mock_result = MagicMock()
        mock_result.content = []
        mock_result.structured_content = {"reward": 0.85}
        mock_result.structuredContent = {"reward": 0.85}
        mock_result.isError = False

        env = self._build_env(is_local=True, mock_result=mock_result)
        result = asyncio.get_event_loop().run_until_complete(
            env._execute_tool("test_tool", {})
        )

        assert result.structuredContent is not None, (
            "_execute_tool dropped structuredContent from local tool result"
        )
        assert result.structuredContent.get("reward") == 0.85

    def test_remote_tool_preserves_structured_content(self):
        """Remote tool results must include structuredContent in the MCPToolResult."""
        from mcp.types import CallToolResult

        mock_result = CallToolResult(
            content=[],
            structuredContent={"reward": 0.95},
            isError=False,
        )

        env = self._build_env(is_local=False, mock_result=mock_result)
        result = asyncio.get_event_loop().run_until_complete(
            env._execute_tool("test_tool", {})
        )

        assert result.structuredContent is not None, (
            "_execute_tool dropped structuredContent from remote tool result"
        )
        assert result.structuredContent.get("reward") == 0.95


# ---------------------------------------------------------------------------
# 2. find_reward correctly extracts rewards from structuredContent
# ---------------------------------------------------------------------------


class TestFindRewardFunctional:
    """find_reward() must extract rewards from structuredContent."""

    def test_finds_reward_in_structured_content(self):
        from hud.agents.base import find_reward
        from hud.types import MCPToolResult

        result = MCPToolResult(
            content=[],
            structuredContent={"reward": 0.85},
        )
        assert find_reward(result) == 0.85

    def test_finds_reward_in_subscores(self):
        from hud.agents.base import find_reward
        from hud.types import MCPToolResult

        result = MCPToolResult(
            content=[],
            structuredContent={
                "subscores": {"test_pass": 0.7, "pr_quality": 0.9},
                "weights": {"test_pass": 0.7, "pr_quality": 0.3},
            },
        )
        reward = find_reward(result)
        expected = 0.7 * 0.7 + 0.9 * 0.3
        assert abs(reward - expected) < 0.01

    def test_returns_zero_when_no_structured_content(self):
        from hud.agents.base import find_reward
        from hud.types import MCPToolResult

        result = MCPToolResult(content=[], structuredContent=None)
        assert find_reward(result) == 0.0

    def test_error_result_returns_zero(self):
        from hud.agents.base import find_reward
        from hud.types import MCPToolResult

        result = MCPToolResult(
            content=[],
            structuredContent={"reward": 1.0},
            isError=True,
        )
        assert find_reward(result) == 0.0


# ---------------------------------------------------------------------------
# 3. Evaluate tool rewards must propagate to final ctx.reward
# ---------------------------------------------------------------------------


class TestEvaluateRewardPropagation:
    """Environment.__aexit__ → _evaluate_reward → EvalContext.reward"""

    def test_environment_aexit_computes_evaluate_reward(self):
        """Environment.__aexit__ must compute _evaluate_reward from evaluate tool calls."""
        from hud.environment.environment import Environment
        from hud.types import MCPToolResult

        env = object.__new__(Environment)
        env._evaluate_calls = [("grade_tool", {"answer": "test"})]
        env._in_context = True
        env._connections = {}
        env._router = MagicMock()
        env._router.clear = MagicMock()
        env._tool_routing_built = True
        env._prompt_routing_built = True

        async def mock_execute(name, args):
            return MCPToolResult(
                content=[],
                structuredContent={"reward": 0.75},
            )

        env._execute_tool = mock_execute

        asyncio.get_event_loop().run_until_complete(
            env.__aexit__(None, None, None)
        )

        assert hasattr(env, "_evaluate_reward"), (
            "Environment.__aexit__ did not set _evaluate_reward"
        )
        assert env._evaluate_reward == 0.75, (
            f"Expected _evaluate_reward=0.75, got {env._evaluate_reward}"
        )

    def test_eval_context_propagates_evaluate_reward(self):
        """EvalContext.__aexit__ must copy _evaluate_reward to self.reward."""
        from hud.eval.context import EvalContext
        from hud.types import MCPToolResult

        ctx = object.__new__(EvalContext)
        ctx.reward = None
        ctx.error = None
        ctx._token = None
        ctx._api_key_token = None
        ctx._evaluate_calls = [("grade", {})]
        ctx._in_context = True
        ctx._connections = {}
        ctx._router = MagicMock()
        ctx._router.clear = MagicMock()
        ctx._tool_routing_built = True
        ctx._prompt_routing_built = True
        ctx.trace_id = "test-trace"
        ctx._scenario_runner = None
        ctx._is_summary = False

        async def mock_execute(name, args):
            return MCPToolResult(
                content=[],
                structuredContent={"reward": 0.85},
            )

        ctx._execute_tool = mock_execute

        async def run():
            with patch("hud.eval.context.flush"), \
                 patch.object(ctx, "_run_task_scenario_evaluate", new_callable=AsyncMock), \
                 patch.object(ctx, "_eval_exit", new_callable=AsyncMock), \
                 patch.object(ctx, "_print_single_result", MagicMock()):
                await ctx.__aexit__(None, None, None)

        asyncio.get_event_loop().run_until_complete(run())

        assert ctx.reward is not None, (
            "EvalContext.__aexit__ did not set self.reward from evaluate tools. "
            "After evaluate tools compute a reward, it must propagate to ctx.reward."
        )
        assert ctx.reward == 0.85, (
            f"Expected ctx.reward=0.85 from evaluate tools, got {ctx.reward}"
        )


# ---------------------------------------------------------------------------
# 4. Evaluate tool reward must win over agent's result.reward
# ---------------------------------------------------------------------------


class TestRunnerRewardHandling:
    """The final ctx.reward must come from evaluate tools, not from the agent run.

    The runner calls agent.run() which returns result.reward (typically 0.0).
    Then EvalContext.__aexit__ runs evaluate tools which compute the real reward.
    Regardless of whether the runner touches ctx.reward or not, the final
    value of ctx.reward after __aexit__ must reflect the evaluate tools.
    """

    def _make_eval_context(self, *, pre_set_reward=None):
        """Build an EvalContext with mocked evaluate tools returning reward=0.85."""
        from hud.eval.context import EvalContext
        from hud.types import MCPToolResult

        ctx = object.__new__(EvalContext)
        ctx.reward = pre_set_reward
        ctx.error = None
        ctx._token = None
        ctx._api_key_token = None
        ctx._evaluate_calls = [("grade", {})]
        ctx._in_context = True
        ctx._connections = {}
        ctx._router = MagicMock()
        ctx._router.clear = MagicMock()
        ctx._tool_routing_built = True
        ctx._prompt_routing_built = True
        ctx.trace_id = "test-trace"
        ctx._scenario_runner = None
        ctx._is_summary = False

        async def mock_execute(name, args):
            return MCPToolResult(
                content=[],
                structuredContent={"reward": 0.85},
            )

        ctx._execute_tool = mock_execute
        return ctx

    def _run_aexit(self, ctx):
        async def run():
            with patch("hud.eval.context.flush"), \
                 patch.object(ctx, "_run_task_scenario_evaluate", new_callable=AsyncMock), \
                 patch.object(ctx, "_eval_exit", new_callable=AsyncMock), \
                 patch.object(ctx, "_print_single_result", MagicMock()):
                await ctx.__aexit__(None, None, None)

        asyncio.get_event_loop().run_until_complete(run())

    def test_evaluate_reward_wins_when_runner_does_not_set_reward(self):
        """If the runner leaves ctx.reward as None, evaluate tools must set it."""
        ctx = self._make_eval_context(pre_set_reward=None)
        self._run_aexit(ctx)

        assert ctx.reward == 0.85, (
            f"ctx.reward should be 0.85 from evaluate tools, got {ctx.reward}. "
            "EvalContext.__aexit__ must propagate _evaluate_reward to self.reward."
        )

    def test_evaluate_reward_wins_over_runner_zero(self):
        """If the runner pre-sets ctx.reward=0.0, evaluate tools must still win.

        This simulates the buggy flow: runner does ctx.reward = result.reward
        (which is 0.0 because the agent didn't produce a reward), then
        __aexit__ runs evaluate tools that compute reward=0.85. The final
        ctx.reward must be 0.85, not 0.0.
        """
        ctx = self._make_eval_context(pre_set_reward=0.0)
        self._run_aexit(ctx)

        assert ctx.reward == 0.85, (
            f"ctx.reward should be 0.85 from evaluate tools, got {ctx.reward}. "
            "When evaluate tools compute a reward, it must take precedence "
            "over any value set by the runner before __aexit__."
        )


# ---------------------------------------------------------------------------
# 5. CLI --full must compose --all, --auto-respond, --max-steps 100
# ---------------------------------------------------------------------------


class TestEvalCLIFlags:
    """--full flag must compose --all, --auto-respond, and --max-steps 100.

    Tests call merge_cli directly and check the resulting config object.
    """

    def test_full_flag_sets_all(self):
        from hud.cli.eval import EvalConfig

        cfg = EvalConfig()
        merged = cfg.merge_cli(full=True)
        assert merged.all is True, "--full did not set all=True"

    def test_full_flag_sets_auto_respond(self):
        from hud.cli.eval import EvalConfig

        cfg = EvalConfig()
        merged = cfg.merge_cli(full=True)
        assert merged.auto_respond is True, (
            "--full did not set auto_respond=True. "
            "The --full flag should compose --all, --auto-respond, and --max-steps 100."
        )

    def test_full_flag_sets_max_steps_100(self):
        from hud.cli.eval import EvalConfig

        cfg = EvalConfig()
        merged = cfg.merge_cli(full=True)
        assert merged.max_steps == 100, (
            f"--full set max_steps={merged.max_steps}, expected 100. "
            "The --full flag should compose --all, --auto-respond, and --max-steps 100."
        )

    def test_full_flag_does_not_override_explicit_overrides(self):
        """User-specified values should take precedence over --full defaults."""
        from hud.cli.eval import EvalConfig

        cfg = EvalConfig()
        merged = cfg.merge_cli(full=True, max_steps=50)
        assert merged.max_steps == 50, (
            "--full should not override an explicitly set max_steps"
        )

    def test_max_steps_default_is_10(self):
        from hud.cli.eval import EvalConfig

        cfg = EvalConfig()
        assert cfg.max_steps == 10, (
            f"max_steps default should be 10, got {cfg.max_steps}"
        )
