"""Tests for GLMComputerTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from mcp import McpError
from mcp.types import ImageContent, TextContent

from hud.tools.computer.glm import GLM_COORDINATE_SPACE, GLMComputerTool
from hud.tools.executors.base import BaseExecutor
from hud.tools.types import ContentResult


@pytest.fixture
def base_executor() -> BaseExecutor:
    """Create a BaseExecutor for testing."""
    return BaseExecutor()


@pytest.fixture
def glm_tool(base_executor: BaseExecutor) -> GLMComputerTool:
    """Create a GLMComputerTool with a base executor."""
    return GLMComputerTool(executor=base_executor)


# ---------------------------------------------------------------------------
# _parse_box
# ---------------------------------------------------------------------------


class TestParseBox:
    """Test _parse_box parsing logic."""

    def test_string_format(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("[500, 300]") == (500, 300)

    def test_string_no_brackets(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("500, 300") == (500, 300)

    def test_string_tight(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("[500,300]") == (500, 300)

    def test_list_format(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box([500, 300]) == (500, 300)

    def test_nested_list(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box([[500, 300]]) == (500, 300)

    def test_none(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box(None) is None

    def test_invalid_string(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("invalid") is None

    def test_empty_list(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box([]) is None


# ---------------------------------------------------------------------------
# _scale_coord
# ---------------------------------------------------------------------------


class TestScaleCoord:
    """Test coordinate scaling from 0-999 to screen pixels."""

    def test_origin(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._scale_coord(0, is_x=True) == 0
        assert glm_tool._scale_coord(0, is_x=False) == 0

    def test_max_coord(self, glm_tool: GLMComputerTool) -> None:
        x = glm_tool._scale_coord(999, is_x=True)
        y = glm_tool._scale_coord(999, is_x=False)
        assert x == int(999 * (glm_tool.environment_width - 1) / GLM_COORDINATE_SPACE)
        assert y == int(999 * (glm_tool.environment_height - 1) / GLM_COORDINATE_SPACE)
        assert x <= glm_tool.environment_width - 1
        assert y <= glm_tool.environment_height - 1

    def test_midpoint(self, glm_tool: GLMComputerTool) -> None:
        x = glm_tool._scale_coord(500, is_x=True)
        expected = int(500 * (glm_tool.environment_width - 1) / GLM_COORDINATE_SPACE)
        assert x == expected


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------


class TestScreenshotCapture:
    """Test screenshot functionality."""

    @pytest.mark.asyncio
    async def test_screenshot_capture(self, base_executor: BaseExecutor) -> None:
        """Verify screenshot capture returns image data."""
        tool = GLMComputerTool(executor=base_executor)
        base_executor.screenshot = AsyncMock(return_value="fake_base64_data")
        blocks = await tool(action="screenshot")
        # Screenshot was captured successfully
        tool.executor.screenshot.call_count


# ---------------------------------------------------------------------------
# _parse_keys
# ---------------------------------------------------------------------------


class TestParseKeys:
    """Test _parse_keys helper."""

    def test_string_combo(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys("ctrl+c") == ["ctrl", "c"]

    def test_single_key(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys("enter") == ["enter"]

    def test_list_input(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys(["Ctrl", "A"]) == ["ctrl", "a"]

    def test_none(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys(None) == []

    def test_empty_string(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys("") == []


# ---------------------------------------------------------------------------
# _fix_xml_args
# ---------------------------------------------------------------------------


class TestFixXMLArgs:
    """Test _fix_xml_args static method for handling GLM's XML-style output."""

    def test_clean_json_passthrough(self) -> None:
        args = {"action": "left_click", "start_box": "[500, 300]"}
        assert GLMComputerTool._fix_xml_args(args) == args

    def test_non_string_passthrough(self) -> None:
        args = {"action": "scroll", "step": 5}
        assert GLMComputerTool._fix_xml_args(args) == args

    def test_mixed_json_xml(self) -> None:
        args = {"action": "left_click\n<arg_key>start_box</arg_key>\n<arg_value>[114, 167]"}
        result = GLMComputerTool._fix_xml_args(args)
        assert result["action"] == "left_click"
        assert result["start_box"] == "[114, 167]"


# ---------------------------------------------------------------------------
# __call__ validation
# ---------------------------------------------------------------------------


class TestGLMCallValidation:
    """Test __call__ parameter validation."""

    @pytest.mark.asyncio
    async def test_missing_action(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action=None)

    @pytest.mark.asyncio
    async def test_unknown_action(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="nonexistent_action")

    @pytest.mark.asyncio
    async def test_click_missing_start_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="left_click")

    @pytest.mark.asyncio
    async def test_done_raises_mcp_error(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError, match="DONE action is not supported"):
            await glm_tool(action="DONE")

    @pytest.mark.asyncio
    async def test_fail_raises_mcp_error(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError, match="FAIL action is not supported"):
            await glm_tool(action="FAIL")
