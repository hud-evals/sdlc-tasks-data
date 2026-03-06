from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from mcp import types

from hud.eval.context import set_trace_context


class FakeFastMCPResult:
    def __init__(self, text: str = "ok") -> None:
        self.content = [types.TextContent(type="text", text=text)]
        self.is_error = False
        self.structured_content = None


class FakeMCPUseResult:
    def __init__(self, text: str = "ok") -> None:
        self.content = [types.TextContent(type="text", text=text)]
        self.isError = False
        self.structuredContent = None


class FakeConnectorResult:
    def __init__(self, text: str = "ok") -> None:
        self.content = [types.TextContent(type="text", text=text)]
        self.isError = False
        self.structuredContent = None


class RecordingFastMCPClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcomes = list(outcomes)

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        raise_on_error: bool = False,
        **kwargs: object,
    ) -> FakeFastMCPResult:
        self.calls.append({"name": name, "arguments": arguments, "kwargs": kwargs})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingMCPUseSession:
    def __init__(self, outcomes: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcomes = list(outcomes)

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> FakeMCPUseResult:
        self.calls.append({"name": name, "arguments": arguments, "kwargs": kwargs})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingConnectorClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcomes = list(outcomes)

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> FakeConnectorResult:
        self.calls.append({"name": name, "arguments": arguments, "kwargs": kwargs})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def is_connected(self) -> bool:
        return True


def _ensure_fake_mcp_use() -> None:
    if "mcp_use.client.client" in sys.modules:
        return

    mcp_use_pkg = ModuleType("mcp_use")
    client_pkg = ModuleType("mcp_use.client")
    client_module = ModuleType("mcp_use.client.client")
    session_module = ModuleType("mcp_use.client.session")

    class MCPClient:  # pragma: no cover - dependency shim for import only
        pass

    class MCPSession:  # pragma: no cover - dependency shim for import only
        pass

    client_module.MCPClient = MCPClient
    session_module.MCPSession = MCPSession
    client_pkg.client = client_module
    client_pkg.session = session_module
    mcp_use_pkg.client = client_pkg

    sys.modules["mcp_use"] = mcp_use_pkg
    sys.modules["mcp_use.client"] = client_pkg
    sys.modules["mcp_use.client.client"] = client_module
    sys.modules["mcp_use.client.session"] = session_module


@pytest.mark.asyncio
async def test_fastmcp_does_not_replay_real_typeerror() -> None:
    from hud.clients.fastmcp import FastMCPHUDClient
    from hud.types import MCPToolCall

    client = FastMCPHUDClient()
    fake_client = RecordingFastMCPClient([TypeError("tool implementation exploded")])
    client._client = fake_client

    with pytest.raises(TypeError, match="tool implementation exploded"):
        with set_trace_context("trace-123"):
            await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["kwargs"] == {"meta": {"_hud_trace_id": "trace-123"}}


@pytest.mark.asyncio
async def test_fastmcp_retries_once_for_unexpected_keyword_argument() -> None:
    from hud.clients.fastmcp import FastMCPHUDClient
    from hud.types import MCPToolCall

    client = FastMCPHUDClient()
    fake_client = RecordingFastMCPClient(
        [TypeError("unexpected keyword argument 'meta'"), FakeFastMCPResult("done")]
    )
    client._client = fake_client

    with set_trace_context("trace-123"):
        result = await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert result.content[0].text == "done"
    assert len(fake_client.calls) == 2
    assert fake_client.calls[0]["kwargs"] == {"meta": {"_hud_trace_id": "trace-123"}}
    assert fake_client.calls[1]["kwargs"] == {}


@pytest.mark.asyncio
async def test_mcp_use_does_not_replay_real_typeerror() -> None:
    _ensure_fake_mcp_use()
    from hud.clients.mcp_use import MCPUseHUDClient
    from hud.types import MCPToolCall

    client = MCPUseHUDClient()
    client._initialized = True
    client._client = object()
    session = RecordingMCPUseSession([TypeError("remote tool exploded")])
    client._tool_map = {"dangerous": ("svc", SimpleNamespace(name="dangerous"), None)}
    client._sessions = {
        "svc": SimpleNamespace(connector=SimpleNamespace(client_session=session))
    }

    with pytest.raises(TypeError, match="remote tool exploded"):
        with set_trace_context("trace-123"):
            await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert len(session.calls) == 1
    assert session.calls[0]["kwargs"] == {"meta": {"_hud_trace_id": "trace-123"}}


@pytest.mark.asyncio
async def test_mcp_use_retries_once_for_unexpected_keyword_argument() -> None:
    _ensure_fake_mcp_use()
    from hud.clients.mcp_use import MCPUseHUDClient
    from hud.types import MCPToolCall

    client = MCPUseHUDClient()
    client._initialized = True
    client._client = object()
    session = RecordingMCPUseSession(
        [TypeError("unexpected keyword argument 'meta'"), FakeMCPUseResult("done")]
    )
    client._tool_map = {"dangerous": ("svc", SimpleNamespace(name="dangerous"), None)}
    client._sessions = {
        "svc": SimpleNamespace(connector=SimpleNamespace(client_session=session))
    }

    with set_trace_context("trace-123"):
        result = await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert result.content[0].text == "done"
    assert len(session.calls) == 2
    assert session.calls[0]["kwargs"] == {"meta": {"_hud_trace_id": "trace-123"}}
    assert session.calls[1]["kwargs"] == {}


@pytest.mark.asyncio
async def test_connector_does_not_replay_real_typeerror() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = RecordingConnectorClient([TypeError("live tool failed")])
    connector.client = client

    with pytest.raises(TypeError, match="live tool failed"):
        with set_trace_context("trace-123"):
            await connector.call_tool("dangerous", {"value": 1})

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"] == {"meta": {"_hud_trace_id": "trace-123"}}


@pytest.mark.asyncio
async def test_connector_retries_once_when_meta_is_unsupported() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = RecordingConnectorClient(
        [TypeError("unexpected keyword argument 'meta'"), FakeConnectorResult("done")]
    )
    connector.client = client

    with set_trace_context("trace-123"):
        result = await connector.call_tool("dangerous", {"value": 1})

    assert result.content[0].text == "done"
    assert len(client.calls) == 2
    assert client.calls[0]["kwargs"] == {"meta": {"_hud_trace_id": "trace-123"}}
    assert client.calls[1]["kwargs"] == {"_meta": {"_hud_trace_id": "trace-123"}}


@pytest.mark.asyncio
async def test_connector_does_not_swallow_real_typeerror_from_meta_fallback() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = RecordingConnectorClient(
        [TypeError("unexpected keyword argument 'meta'"), TypeError("remote tool exploded")]
    )
    connector.client = client

    with pytest.raises(TypeError, match="remote tool exploded"):
        with set_trace_context("trace-123"):
            await connector.call_tool("dangerous", {"value": 1})

    assert len(client.calls) == 2
    assert client.calls[0]["kwargs"] == {"meta": {"_hud_trace_id": "trace-123"}}
    assert client.calls[1]["kwargs"] == {"_meta": {"_hud_trace_id": "trace-123"}}


@pytest.mark.asyncio
async def test_connector_falls_back_to_plain_arguments_when_meta_and__meta_are_unsupported() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = RecordingConnectorClient(
        [
            TypeError("unexpected keyword argument 'meta'"),
            TypeError("unexpected keyword argument '_meta'"),
            FakeConnectorResult("done"),
        ]
    )
    connector.client = client

    with set_trace_context("trace-123"):
        result = await connector.call_tool("dangerous", {"value": 1})

    assert result.content[0].text == "done"
    assert len(client.calls) == 3
    assert client.calls[2]["kwargs"] == {}
