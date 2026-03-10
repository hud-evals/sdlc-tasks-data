from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from mcp import types

from hud.eval.context import set_trace_context


TRACE_META = {"_hud_trace_id": "trace-123"}


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


class FastMCPMetaClient:
    def __init__(self, outcome: object | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0
        self.outcome = outcome if outcome is not None else FakeFastMCPResult("done")

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        raise_on_error: bool = False,
        meta: dict[str, object] | None = None,
    ) -> FakeFastMCPResult:
        self.calls.append({"name": name, "arguments": arguments, "meta": meta})
        self.side_effect_count += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FastMCPPlainClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        raise_on_error: bool = False,
    ) -> FakeFastMCPResult:
        self.calls.append({"name": name, "arguments": arguments})
        self.side_effect_count += 1
        return FakeFastMCPResult("done")


class FastMCPKwargsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        raise_on_error: bool = False,
        **kwargs: object,
    ) -> FakeFastMCPResult:
        self.calls.append({"name": name, "arguments": arguments, "kwargs": kwargs})
        self.side_effect_count += 1
        return FakeFastMCPResult("done")


class MCPUseMetaSession:
    def __init__(self, outcome: object | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0
        self.outcome = outcome if outcome is not None else FakeMCPUseResult("done")

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        meta: dict[str, object] | None = None,
    ) -> FakeMCPUseResult:
        self.calls.append({"name": name, "arguments": arguments, "meta": meta})
        self.side_effect_count += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class MCPUsePlainSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
    ) -> FakeMCPUseResult:
        self.calls.append({"name": name, "arguments": arguments})
        self.side_effect_count += 1
        return FakeMCPUseResult("done")


class MCPUseKwargsSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> FakeMCPUseResult:
        self.calls.append({"name": name, "arguments": arguments, "kwargs": kwargs})
        self.side_effect_count += 1
        return FakeMCPUseResult("done")


class ConnectorMetaClient:
    def __init__(self, outcome: object | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0
        self.outcome = outcome if outcome is not None else FakeConnectorResult("done")

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        meta: dict[str, object] | None = None,
    ) -> FakeConnectorResult:
        self.calls.append({"name": name, "arguments": arguments, "meta": meta})
        self.side_effect_count += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def is_connected(self) -> bool:
        return True


class ConnectorUnderscoreMetaClient:
    def __init__(self, outcome: object | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0
        self.outcome = outcome if outcome is not None else FakeConnectorResult("done")

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        _meta: dict[str, object] | None = None,
    ) -> FakeConnectorResult:
        self.calls.append({"name": name, "arguments": arguments, "_meta": _meta})
        self.side_effect_count += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def is_connected(self) -> bool:
        return True


class ConnectorPlainClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
    ) -> FakeConnectorResult:
        self.calls.append({"name": name, "arguments": arguments})
        self.side_effect_count += 1
        return FakeConnectorResult("done")

    def is_connected(self) -> bool:
        return True


class ConnectorKwargsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.side_effect_count = 0

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> FakeConnectorResult:
        self.calls.append({"name": name, "arguments": arguments, "kwargs": kwargs})
        self.side_effect_count += 1
        return FakeConnectorResult("done")

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


def _build_mcp_use_client(session: object):
    from hud.clients.mcp_use import MCPUseHUDClient

    client = MCPUseHUDClient()
    client._initialized = True
    client._client = object()
    client._tool_map = {"dangerous": ("svc", SimpleNamespace(name="dangerous"), None)}
    client._sessions = {
        "svc": SimpleNamespace(connector=SimpleNamespace(client_session=session))
    }
    return client


@pytest.mark.asyncio
async def test_fastmcp_propagates_real_typeerror_without_replay() -> None:
    from hud.clients.fastmcp import FastMCPHUDClient
    from hud.types import MCPToolCall

    client = FastMCPHUDClient()
    fake_client = FastMCPMetaClient(TypeError("tool implementation exploded"))
    client._client = fake_client

    with pytest.raises(TypeError, match="tool implementation exploded"):
        with set_trace_context("trace-123"):
            await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert fake_client.side_effect_count == 1
    assert fake_client.calls[0]["meta"] == TRACE_META


@pytest.mark.asyncio
async def test_fastmcp_preserves_compatibility_without_meta_support() -> None:
    from hud.clients.fastmcp import FastMCPHUDClient
    from hud.types import MCPToolCall

    client = FastMCPHUDClient()
    fake_client = FastMCPPlainClient()
    client._client = fake_client

    with set_trace_context("trace-123"):
        result = await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert result.content[0].text == "done"
    assert fake_client.side_effect_count == 1


@pytest.mark.asyncio
async def test_fastmcp_still_propagates_meta_when_kwargs_supports_it() -> None:
    from hud.clients.fastmcp import FastMCPHUDClient
    from hud.types import MCPToolCall

    client = FastMCPHUDClient()
    fake_client = FastMCPKwargsClient()
    client._client = fake_client

    with set_trace_context("trace-123"):
        await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert fake_client.side_effect_count == 1
    assert fake_client.calls[0]["kwargs"].get("meta") == TRACE_META


@pytest.mark.asyncio
async def test_mcp_use_propagates_real_typeerror_without_replay() -> None:
    _ensure_fake_mcp_use()
    from hud.types import MCPToolCall

    session = MCPUseMetaSession(TypeError("remote tool exploded"))
    client = _build_mcp_use_client(session)

    with pytest.raises(TypeError, match="remote tool exploded"):
        with set_trace_context("trace-123"):
            await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert session.side_effect_count == 1
    assert session.calls[0]["meta"] == TRACE_META


@pytest.mark.asyncio
async def test_mcp_use_preserves_compatibility_without_meta_support() -> None:
    _ensure_fake_mcp_use()
    from hud.types import MCPToolCall

    session = MCPUsePlainSession()
    client = _build_mcp_use_client(session)

    with set_trace_context("trace-123"):
        result = await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert result.content[0].text == "done"
    assert session.side_effect_count == 1


@pytest.mark.asyncio
async def test_mcp_use_still_propagates_meta_when_kwargs_supports_it() -> None:
    _ensure_fake_mcp_use()
    from hud.types import MCPToolCall

    session = MCPUseKwargsSession()
    client = _build_mcp_use_client(session)

    with set_trace_context("trace-123"):
        await client._call_tool(MCPToolCall(name="dangerous", arguments={"value": 1}))

    assert session.side_effect_count == 1
    assert session.calls[0]["kwargs"].get("meta") == TRACE_META


@pytest.mark.asyncio
async def test_connector_propagates_real_typeerror_without_replay() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = ConnectorMetaClient(TypeError("live tool failed"))
    connector.client = client

    with pytest.raises(TypeError, match="live tool failed"):
        with set_trace_context("trace-123"):
            await connector.call_tool("dangerous", {"value": 1})

    assert client.side_effect_count == 1
    assert client.calls[0]["meta"] == TRACE_META


@pytest.mark.asyncio
async def test_connector_preserves__meta_compatibility() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = ConnectorUnderscoreMetaClient()
    connector.client = client

    with set_trace_context("trace-123"):
        result = await connector.call_tool("dangerous", {"value": 1})

    assert result.content[0].text == "done"
    assert client.side_effect_count == 1
    assert client.calls[0]["_meta"] == TRACE_META


@pytest.mark.asyncio
async def test_connector_propagates_real_typeerror_from__meta_path_without_replay() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = ConnectorUnderscoreMetaClient(TypeError("remote tool exploded"))
    connector.client = client

    with pytest.raises(TypeError, match="remote tool exploded"):
        with set_trace_context("trace-123"):
            await connector.call_tool("dangerous", {"value": 1})

    assert client.side_effect_count == 1
    assert client.calls[0]["_meta"] == TRACE_META


@pytest.mark.asyncio
async def test_connector_preserves_compatibility_without_meta_or__meta_support() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = ConnectorPlainClient()
    connector.client = client

    with set_trace_context("trace-123"):
        result = await connector.call_tool("dangerous", {"value": 1})

    assert result.content[0].text == "done"
    assert client.side_effect_count == 1


@pytest.mark.asyncio
async def test_connector_still_propagates_meta_when_kwargs_supports_it() -> None:
    from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

    connector = Connector(
        transport=None,
        config=ConnectionConfig(),
        name="svc",
        connection_type=ConnectionType.REMOTE,
    )
    client = ConnectorKwargsClient()
    connector.client = client

    with set_trace_context("trace-123"):
        await connector.call_tool("dangerous", {"value": 1})

    assert client.side_effect_count == 1
    assert client.calls[0]["kwargs"].get("meta") == TRACE_META
