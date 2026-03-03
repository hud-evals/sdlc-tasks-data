from __future__ import annotations

import pytest

import mcp.types as mcp_types
from hud.types import MCPToolResult


class _FailingConnectConnector:
    def __init__(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def cached_tools(self) -> list[mcp_types.Tool]:
        return []

    @property
    def cached_prompts(self) -> list[mcp_types.Prompt]:
        return []

    @property
    def cached_resources(self) -> list[mcp_types.Resource]:
        return []

    async def connect(self) -> None:
        raise RuntimeError("dial failed")

    async def disconnect(self) -> None:
        self._connected = False

    async def list_tools(self) -> list[mcp_types.Tool]:
        return []

    async def list_prompts(self) -> list[mcp_types.Prompt]:
        return []

    async def list_resources(self) -> list[mcp_types.Resource]:
        return []


class TestEnvironmentConnectionState:
    @pytest.mark.asyncio
    async def test_is_connected_false_after_connect_failure(self) -> None:
        from hud.environment import Environment

        env = Environment("test")
        env._connections["broken"] = _FailingConnectConnector()  # type: ignore[assignment]

        with pytest.raises(ConnectionError, match="Failed to connect to broken"):
            await env.__aenter__()

        assert env.is_connected is False

    @pytest.mark.asyncio
    async def test_is_connected_false_after_setup_tool_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hud.environment import Environment

        env = Environment("test")
        env.setup_tool("failing_setup")

        async def _fake_execute_tool(name: str, arguments: dict[str, object]) -> MCPToolResult:
            assert name == "failing_setup"
            _ = arguments
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text="setup crashed")],
                isError=True,
            )

        monkeypatch.setattr(env, "_execute_tool", _fake_execute_tool)

        with pytest.raises(RuntimeError, match="Setup tool 'failing_setup' failed: setup crashed"):
            await env.__aenter__()

        assert env.is_connected is False

    @pytest.mark.asyncio
    async def test_is_connected_true_inside_context_and_false_after_exit(self) -> None:
        from hud.environment import Environment

        env = Environment("test")
        assert env.is_connected is False

        async with env:
            assert env.is_connected is True

        assert env.is_connected is False
