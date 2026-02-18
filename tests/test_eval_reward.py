"""Integration tests for the eval reward pipeline.

These tests verify that rewards flow correctly from tool results
through the eval context to the final score, and that CLI flags
compose correctly.

The tests use source-code inspection to validate behavior without
depending on the installed package version, ensuring they work
correctly in the grading environment.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap

import pytest


def _read(relpath: str) -> str:
    """Read a source file relative to the repo root."""
    return open(relpath).read()


# ---------------------------------------------------------------------------
# 1. structuredContent must pass through _execute_tool
# ---------------------------------------------------------------------------


class TestStructuredContentPassthrough:
    """_execute_tool must forward structuredContent from tool results."""

    def test_local_tool_passes_structured_content(self):
        source = _read("hud/environment/environment.py")

        local_block = source[source.find("is_local(name)") :]
        local_block = local_block[: local_block.find("connection_name")]

        assert "structured_content" in local_block or "structuredContent" in local_block, (
            "_execute_tool drops structuredContent for local tools. "
            "The MCPToolResult constructor must include "
            "structuredContent=result.structured_content"
        )

    def test_remote_tool_passes_structured_content(self):
        source = _read("hud/environment/environment.py")

        remote_block = source[source.find("get_connection(name)") :]
        remote_block = remote_block[: remote_block.find("raise ValueError")]

        assert "structuredContent" in remote_block, (
            "_execute_tool drops structuredContent for remote tools. "
            "The MCPToolResult constructor must include "
            "structuredContent=result.structuredContent"
        )


# ---------------------------------------------------------------------------
# 2. EvalContext must propagate _evaluate_reward to self.reward
# ---------------------------------------------------------------------------


class TestEvalContextReward:
    """EvalContext.__aexit__ must copy _evaluate_reward to self.reward."""

    def test_evaluate_reward_propagation(self):
        source = _read("hud/eval/context.py")

        assert "self.reward = self._evaluate_reward" in source, (
            "EvalContext.__aexit__ does not set self.reward from _evaluate_reward. "
            "Add: if self.reward is None and hasattr(self, '_evaluate_reward'): "
            "self.reward = self._evaluate_reward"
        )


# ---------------------------------------------------------------------------
# 3. CLI --full must compose --all, --auto-respond, --max-steps 100
# ---------------------------------------------------------------------------


class TestEvalCLIFlags:
    """--full flag must compose --all, --auto-respond, and --max-steps 100."""

    def _get_merge_cli_source(self) -> str:
        source = _read("hud/cli/eval.py")
        start = source.find("def merge_cli")
        end = source.find("\n    def ", start + 1)
        return source[start:end]

    def test_full_sets_all(self):
        src = self._get_merge_cli_source()

        has_full_to_all = (
            'overrides["all"] = True' in src
            or "overrides['all'] = True" in src
        )
        assert has_full_to_all, (
            "--full must set all=True in merge_cli. "
            'Expected: overrides["all"] = True'
        )

    def test_full_sets_auto_respond(self):
        src = self._get_merge_cli_source()

        has_auto_respond = (
            'overrides["auto_respond"] = True' in src
            or "overrides['auto_respond'] = True" in src
            or "auto_respond" in src
        )

        full_block = src[src.find("full") :]
        sets_auto_respond = "auto_respond" in full_block and "True" in full_block

        assert sets_auto_respond, (
            "--full must set auto_respond=True. "
            "Currently --full only sets all=True, missing auto_respond."
        )

    def test_full_sets_max_steps(self):
        src = self._get_merge_cli_source()

        full_block = src[src.find("full") :]
        sets_max_steps = "max_steps" in full_block and "100" in full_block

        assert sets_max_steps, (
            "--full must set max_steps=100. "
            "Currently --full does not set max_steps."
        )

    def test_max_steps_default_is_10(self):
        source = _read("hud/cli/eval.py")

        match = re.search(r'max_steps:\s*int\s*=\s*(\d+)', source)
        assert match is not None, "Could not find max_steps default in EvalConfig"
        assert match.group(1) == "10", (
            f"max_steps default should be 10, got {match.group(1)}"
        )


# ---------------------------------------------------------------------------
# 4. Runner must NOT override context reward
# ---------------------------------------------------------------------------


class TestRunnerRewardHandling:
    """Runner must not clobber ctx.reward with result.reward."""

    def test_run_dataset_does_not_override_ctx_reward(self):
        source = _read("hud/datasets/runner.py")

        assert "ctx.reward = result.reward" not in source, (
            "run_dataset sets ctx.reward = result.reward, which overwrites "
            "the reward computed by EvalContext.__aexit__ from evaluate tools. "
            "Remove this line — reward propagation is handled by EvalContext."
        )

    def test_run_single_task_does_not_override_ctx_reward(self):
        source = _read("hud/datasets/runner.py")

        assert "ctx.reward = result.reward" not in source, (
            "run_single_task sets ctx.reward = result.reward, which overwrites "
            "the reward computed by EvalContext.__aexit__ from evaluate tools. "
            "Remove this line — reward propagation is handled by EvalContext."
        )


# ---------------------------------------------------------------------------
# 5. find_reward error logging should show structuredContent, not full result
# ---------------------------------------------------------------------------


class TestRewardLogging:
    """find_reward error logging must be actionable."""

    def test_error_log_shows_structured_content_not_full_result(self):
        source = _read("hud/agents/base.py")

        find_reward_src = source[source.find("def find_reward") :]
        next_def = find_reward_src.find("\ndef ", 1)
        if next_def > 0:
            find_reward_src = find_reward_src[:next_def]

        assert "str(result.structuredContent)" in find_reward_src, (
            "find_reward logs the entire MCPToolResult object on parse failure "
            "(logger.error('...', result)), which produces huge unreadable output. "
            "Use str(result.structuredContent) instead."
        )


# ---------------------------------------------------------------------------
# 6. Functional test: find_reward parses structuredContent correctly
# ---------------------------------------------------------------------------


class TestFindRewardFunctional:
    """find_reward() must extract rewards from structuredContent."""

    def _import_find_reward(self):
        """Import find_reward, handling potential import path issues."""
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "hud.agents.base_test", "hud/agents/base.py"
        )
        if spec is None or spec.loader is None:
            pytest.skip("Cannot load hud/agents/base.py as module")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pytest.skip("Cannot execute hud/agents/base.py (missing deps)")
        return mod.find_reward

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
