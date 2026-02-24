"""Integration tests for multi-model native tool spec resolution.

Verifies that NativeToolSpecs list format is properly resolved, that each
agent type correctly handles model-specific spec selection, and that the
AnthropicComputerTool produces correct API parameters for different models.
"""
from __future__ import annotations
import ast
import inspect
from unittest.mock import MagicMock


class TestGetNativeSpecHandlesList:
    """base.py get_native_spec() must handle both single and list specs."""

    def test_returns_spec_for_matching_model(self):
        """When specs is a list, return the one matching the model."""
        from hud.tools.native_types import NativeToolSpec
        from hud.tools.base import BaseTool

        try:
            from hud.types import AgentType
        except ImportError:
            from hud.agents.types import AgentType

        spec_new = NativeToolSpec(
            api_type="computer_20251124",
            api_name="computer",
            beta="computer-use-2025-11-24",
            role="computer",
            supported_models=("claude-opus-4-5*", "claude-opus-4-6*"),
        )
        spec_old = NativeToolSpec(
            api_type="computer_20250124",
            api_name="computer",
            beta="computer-use-2025-01-24",
            role="computer",
        )

        class DummyTool(BaseTool):
            name = "test_tool"
            async def __call__(self, **kwargs):
                return MagicMock()

        tool = DummyTool.__new__(DummyTool)
        tool._native_specs = {AgentType.CLAUDE: [spec_new, spec_old]}

        result = tool.get_native_spec(AgentType.CLAUDE, model="claude-opus-4-6")
        assert result is not None, "get_native_spec returned None for opus 4.6"
        assert result.api_type == "computer_20251124", (
            f"Opus 4.6 should get computer_20251124 but got {result.api_type}"
        )

    def test_returns_fallback_for_non_matching_model(self):
        """When model doesn't match any supported_models, return fallback spec."""
        from hud.tools.native_types import NativeToolSpec

        try:
            from hud.types import AgentType
        except ImportError:
            from hud.agents.types import AgentType

        from hud.tools.base import BaseTool

        spec_new = NativeToolSpec(
            api_type="computer_20251124",
            api_name="computer",
            beta="computer-use-2025-11-24",
            role="computer",
            supported_models=("claude-opus-4-5*", "claude-opus-4-6*"),
        )
        spec_old = NativeToolSpec(
            api_type="computer_20250124",
            api_name="computer",
            beta="computer-use-2025-01-24",
            role="computer",
        )

        class DummyTool(BaseTool):
            name = "test_tool"
            async def __call__(self, **kwargs):
                return MagicMock()

        tool = DummyTool.__new__(DummyTool)
        tool._native_specs = {AgentType.CLAUDE: [spec_new, spec_old]}

        result = tool.get_native_spec(AgentType.CLAUDE, model="claude-sonnet-4-5")
        assert result is not None, "get_native_spec returned None for sonnet 4.5"
        assert result.api_type == "computer_20250124", (
            f"Sonnet 4.5 should get computer_20250124 but got {result.api_type}"
        )

    def test_accepts_model_parameter(self):
        """get_native_spec() must accept an optional model parameter."""
        from hud.tools.base import BaseTool
        sig = inspect.signature(BaseTool.get_native_spec)
        params = list(sig.parameters.keys())
        assert "model" in params, (
            f"get_native_spec() must accept a 'model' parameter. "
            f"Current parameters: {params}"
        )


class TestNativeToolSpecSupportsModel:
    """NativeToolSpec.supports_model() must handle glob patterns."""

    def test_wildcard_match(self):
        from hud.tools.native_types import NativeToolSpec
        spec = NativeToolSpec(
            api_type="computer_20251124",
            api_name="computer",
            supported_models=("claude-opus-4-5*", "claude-opus-4-6*"),
        )
        assert spec.supports_model("claude-opus-4-6") is True
        assert spec.supports_model("claude-opus-4-6-20250610") is True
        assert spec.supports_model("claude-sonnet-4-5") is False

    def test_no_supported_models_matches_all(self):
        from hud.tools.native_types import NativeToolSpec
        spec = NativeToolSpec(
            api_type="computer_20250124",
            api_name="computer",
        )
        assert spec.supports_model("claude-opus-4-6") is True
        assert spec.supports_model("claude-sonnet-4-5") is True
        assert spec.supports_model(None) is True
