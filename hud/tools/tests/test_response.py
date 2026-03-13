"""Tests for ResponseTool class."""

from __future__ import annotations

import pytest

from hud.tools.response import ResponseTool


class ConcreteResponseTool(ResponseTool):
    """Concrete implementation for testing."""

    async def __call__(self, response: str | None = None, messages=None):
        """Concrete implementation."""
        from mcp.types import TextContent

        return [TextContent(text=response or "test", type="text")]


class TestResponseTool:
    """Tests for ResponseTool abstract class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        tool = ConcreteResponseTool()
        assert tool.name == "response"
        assert tool.title == "Response Tool"
        assert tool.description == "Send a text response or list of messages to the environment"

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        tool = ConcreteResponseTool(
            name="custom_response", title="Custom Response Tool", description="Custom description"
        )
        assert tool.name == "custom_response"
        assert tool.title == "Custom Response Tool"
        assert tool.description == "Custom description"

    def test_abstract_method_not_implemented(self):
        """Test that abstract method raises NotImplementedError when not implemented."""

        # Create a concrete tool to test the abstract method's NotImplementedError
        tool = ConcreteResponseTool()

        # This should trigger the NotImplementedError in the abstract method
        with pytest.raises(NotImplementedError, match="Subclasses must implement __call__"):
            # Call the parent abstract method directly to hit the raise line
            import asyncio

            asyncio.run(ResponseTool.__call__(tool, "test"))  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""
        tool = ConcreteResponseTool()
        result = await tool("Hello, World!")

        assert len(result) == 1
        assert result[0].text == "Hello, World!"
        assert result[0].type == "text"
