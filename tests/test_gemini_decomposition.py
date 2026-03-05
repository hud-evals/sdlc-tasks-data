"""Tests for the Gemini agent decomposition — GeminiCUAAgent registration and base class hooks."""

from unittest.mock import MagicMock


class TestGeminiCUARegistration:
    """4 tests: GeminiCUAAgent must be importable and registered."""

    def test_import_from_agents_package(self):
        """GeminiCUAAgent must be importable from hud.agents."""
        from hud.agents import GeminiCUAAgent
        assert GeminiCUAAgent is not None

    def test_agent_type_enum_exists(self):
        """AgentType must have a GEMINI_CUA member."""
        from hud.types import AgentType
        assert hasattr(AgentType, 'GEMINI_CUA')

    def test_agent_type_maps_to_cua_class(self):
        """AgentType.GEMINI_CUA.cls must resolve to GeminiCUAAgent."""
        from hud.types import AgentType
        from hud.agents.gemini_cua import GeminiCUAAgent
        assert AgentType.GEMINI_CUA.cls == GeminiCUAAgent

    def test_cua_inherits_from_gemini(self):
        """GeminiCUAAgent must be a subclass of GeminiAgent."""
        from hud.agents.gemini import GeminiAgent
        from hud.agents.gemini_cua import GeminiCUAAgent
        assert issubclass(GeminiCUAAgent, GeminiAgent)


class TestGeminiBaseDecomposition:
    """4 tests: GeminiAgent must have hooks and no CUA-specific code."""

    def test_has_extract_tool_call_method(self):
        """GeminiAgent must define _extract_tool_call (hook for CUA override)."""
        from hud.agents.gemini import GeminiAgent
        assert hasattr(GeminiAgent, '_extract_tool_call')
        assert callable(getattr(GeminiAgent, '_extract_tool_call'))

    def test_has_to_gemini_tool_method(self):
        """GeminiAgent must define _to_gemini_tool (hook for CUA override)."""
        from hud.agents.gemini import GeminiAgent
        assert hasattr(GeminiAgent, '_to_gemini_tool')
        assert callable(getattr(GeminiAgent, '_to_gemini_tool'))

    def test_metadata_not_display_dimensions(self):
        """GeminiAgent.metadata must not have display dimensions (CUA-only)."""
        from hud.agents.gemini import GeminiAgent
        meta = GeminiAgent.metadata
        if meta is not None:
            assert 'display_width' not in meta, (
                "display_width belongs in GeminiCUAAgent.metadata, "
                "not the base GeminiAgent"
            )

    def test_config_no_excluded_predefined_functions(self):
        """GeminiConfig must not have excluded_predefined_functions (CUA-only)."""
        from hud.agents.gemini import GeminiConfig
        fields = GeminiConfig.model_fields
        assert 'excluded_predefined_functions' not in fields, (
            "excluded_predefined_functions belongs in GeminiCUAConfig, "
            "not GeminiConfig"
        )


class TestHookMethodBehavior:
    """3 tests: hook methods work correctly for standard tool-calling."""

    def test_to_gemini_tool_creates_function_declaration(self):
        """_to_gemini_tool must convert an MCP tool to a function declaration."""
        from hud.agents.gemini import GeminiAgent
        import mcp.types as types

        agent = GeminiAgent.__new__(GeminiAgent)
        agent._gemini_to_mcp_tool_map = {}

        tool = types.Tool(
            name="calculator",
            description="A calculator tool",
            inputSchema={
                "type": "object",
                "properties": {"op": {"type": "string"}},
            },
        )

        result = agent._to_gemini_tool(tool)
        assert result is not None
        fd = getattr(result, 'function_declarations', None)
        assert fd is not None and len(fd) > 0, (
            "_to_gemini_tool should create a function declaration for standard tools"
        )

    def test_extract_tool_call_returns_mcp_tool_call(self):
        """_extract_tool_call must convert a function call part to MCPToolCall."""
        from hud.agents.gemini import GeminiAgent

        agent = GeminiAgent.__new__(GeminiAgent)
        agent._gemini_to_mcp_tool_map = {"calculator": "calculator"}

        part = MagicMock()
        part.function_call = MagicMock()
        part.function_call.name = "calculator"
        part.function_call.args = {"op": "add", "a": 1, "b": 2}

        result = agent._extract_tool_call(part)
        assert result is not None
        assert result.name == "calculator"
        assert result.arguments == {"op": "add", "a": 1, "b": 2}

    def test_extract_tool_call_returns_none_without_function_call(self):
        """_extract_tool_call must return None when part has no function_call."""
        from hud.agents.gemini import GeminiAgent

        agent = GeminiAgent.__new__(GeminiAgent)
        agent._gemini_to_mcp_tool_map = {}

        part = MagicMock()
        part.function_call = None

        result = agent._extract_tool_call(part)
        assert result is None
