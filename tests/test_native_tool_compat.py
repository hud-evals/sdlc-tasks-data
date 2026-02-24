"""Integration tests for multi-model native tool spec resolution.

Verifies that NativeToolSpecs list format is properly resolved, that each
agent type correctly handles model-specific spec selection, and that the
AnthropicComputerTool produces correct API parameters for different models.
"""
from __future__ import annotations
import ast
import inspect
from types import SimpleNamespace
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

    def test_uses_supported_models_for_non_opus_models(self):
        """Selection must be based on supported_models, not model name heuristics."""
        from hud.tools.native_types import NativeToolSpec
        from hud.tools.base import BaseTool

        try:
            from hud.types import AgentType
        except ImportError:
            from hud.agents.types import AgentType

        spec_gpt = NativeToolSpec(
            api_type="computer_gpt",
            api_name="computer",
            supported_models=("gpt-5*",),
        )
        spec_fallback = NativeToolSpec(
            api_type="computer_fallback",
            api_name="computer",
        )

        class DummyTool(BaseTool):
            name = "test_tool"
            async def __call__(self, **kwargs):
                return MagicMock()

        tool = DummyTool.__new__(DummyTool)
        tool._native_specs = {AgentType.CLAUDE: [spec_gpt, spec_fallback]}

        result = tool.get_native_spec(AgentType.CLAUDE, model="gpt-5.1")
        assert result is not None, "get_native_spec returned None for gpt-5.1"
        assert result.api_type == "computer_gpt", (
            f"gpt-5.1 should get computer_gpt but got {result.api_type}"
        )

    def test_returns_none_when_no_spec_matches_and_no_fallback(self):
        """If no spec supports the model and no unrestricted fallback exists, return None."""
        from hud.tools.native_types import NativeToolSpec
        from hud.tools.base import BaseTool

        try:
            from hud.types import AgentType
        except ImportError:
            from hud.agents.types import AgentType

        spec_gpt = NativeToolSpec(
            api_type="computer_gpt",
            api_name="computer",
            supported_models=("gpt-5*",),
        )
        spec_opus = NativeToolSpec(
            api_type="computer_opus",
            api_name="computer",
            supported_models=("claude-opus-4-6*",),
        )

        class DummyTool(BaseTool):
            name = "test_tool"
            async def __call__(self, **kwargs):
                return MagicMock()

        tool = DummyTool.__new__(DummyTool)
        tool._native_specs = {AgentType.CLAUDE: [spec_gpt, spec_opus]}

        result = tool.get_native_spec(AgentType.CLAUDE, model="claude-sonnet-4-5")
        assert result is None, (
            "Expected None when no list entry matches and no fallback spec exists"
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


class TestResolveNativeSpecListSelection:
    """Agent-level resolution should honor list-of-specs model matching."""

    def _make_dummy_agent(self, model: str):
        from hud.agents.base import MCPAgent
        from hud.types import AgentResponse, AgentType

        class DummyAgent(MCPAgent):
            @classmethod
            def agent_type(cls) -> AgentType:
                return AgentType.CLAUDE

            async def get_system_messages(self):
                return []

            async def get_response(self, messages):
                return AgentResponse(content="", tool_calls=[], done=True)

            async def format_blocks(self, blocks):
                return []

            async def format_tool_results(self, tool_calls, tool_results):
                return []

        agent = DummyAgent.__new__(DummyAgent)
        agent.model = model
        return agent

    def test_resolve_native_spec_selects_matching_variant(self):
        agent = self._make_dummy_agent("claude-opus-4-6")
        tool = SimpleNamespace(
            name="computer",
            meta={
                "native_tools": {
                    "claude": [
                        {
                            "api_type": "computer_20251124",
                            "api_name": "computer",
                            "supported_models": ["claude-opus-4-5*", "claude-opus-4-6*"],
                        },
                        {
                            "api_type": "computer_20250124",
                            "api_name": "computer",
                        },
                    ]
                }
            },
        )
        spec = agent.resolve_native_spec(tool)
        assert spec is not None
        assert spec.api_type == "computer_20251124"

    def test_resolve_native_spec_returns_none_without_match_or_fallback(self):
        agent = self._make_dummy_agent("claude-sonnet-4-5")
        tool = SimpleNamespace(
            name="computer",
            meta={
                "native_tools": {
                    "claude": [
                        {
                            "api_type": "computer_20251124",
                            "api_name": "computer",
                            "supported_models": ["claude-opus-4-6*"],
                        }
                    ]
                }
            },
        )
        spec = agent.resolve_native_spec(tool)
        assert spec is None
