from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import mcp.types as mcp_types
import pytest
from fastmcp.client.transports import SSETransport, StreamableHttpTransport

from hud.environment.connection import ConnectionConfig, ConnectionType, Connector


class TestStreamableHttpContainment:
    @pytest.mark.asyncio
    async def test_hub_backed_streamable_http_fails_closed_before_dispatch(self) -> None:
        connector = Connector(
            transport=StreamableHttpTransport(
                url="https://mcp.hud.ai/browser",
                headers={"Environment-Name": "browser", "Environment-Id": "env-123"},
            ),
            config=ConnectionConfig(),
            name="hud",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(side_effect=AssertionError("should not dispatch"))
        connector.client = mock_client

        try:
            result = await connector.call_tool("navigate", {"url": "https://hud.ai"})
        except RuntimeError as exc:
            message = str(exc)
            assert "streamable-http" in message
            assert "temporarily disabled" in message
        else:
            assert result.isError is True
            joined = " ".join(
                getattr(item, "text", "") for item in getattr(result, "content", [])
            )
            assert "streamable-http" in joined
            assert "temporarily" in joined

        mock_client.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_generic_remote_streamable_http_path_is_not_contained(self) -> None:
        connector = Connector(
            transport=StreamableHttpTransport(url="https://mcp.example.com/browser"),
            config=ConnectionConfig(),
            name="external",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(
            return_value=mcp_types.CallToolResult(content=[], isError=False)
        )
        connector.client = mock_client

        await connector.call_tool("search", {"query": "logs"})

        mock_client.call_tool.assert_called_once_with(
            name="search",
            arguments={"query": "logs"},
        )

    @pytest.mark.asyncio
    async def test_remote_sse_path_is_not_contained(self) -> None:
        connector = Connector(
            transport=SSETransport(url="https://mcp.hud.ai/browser"),
            config=ConnectionConfig(prefix="browser"),
            name="hud",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(
            return_value=mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text="ok")],
                isError=False,
            )
        )
        connector.client = mock_client

        result = await connector.call_tool("browser_navigate", {"url": "https://hud.ai"})

        mock_client.call_tool.assert_called_once_with(
            name="navigate",
            arguments={"url": "https://hud.ai"},
        )
        assert result.isError is False
        assert result.content[0].text == "ok"

    @pytest.mark.asyncio
    async def test_local_connector_path_is_not_contained(self) -> None:
        connector = Connector(
            transport={"command": "python", "args": ["-m", "server"]},
            config=ConnectionConfig(),
            name="local",
            connection_type=ConnectionType.LOCAL,
        )

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(
            return_value=mcp_types.CallToolResult(content=[], isError=False)
        )
        connector.client = mock_client

        await connector.call_tool("search", {"query": "logs"})

        mock_client.call_tool.assert_called_once_with(
            name="search",
            arguments={"query": "logs"},
        )
