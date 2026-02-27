"""Hidden regression tests for parallel context header isolation."""

from __future__ import annotations

from types import SimpleNamespace

from hud.environment import Environment
from hud.environment.connection import ConnectionConfig, ConnectionType, Connector
from hud.eval.context import EvalContext


def test_connector_copy_isolates_transport_and_config() -> None:
    """Connector.copy should not share mutable transport/config state."""
    transport = {
        "url": "https://mcp.hud.so/jsonrpc",
        "headers": {"Environment-Name": "browser", "Environment-Id": "env-1"},
    }
    connector = Connector(
        transport=transport,
        config=ConnectionConfig(include=["tool1"], exclude=["tool2"]),
        name="hud",
        connection_type=ConnectionType.REMOTE,
    )

    copied = connector.copy()

    assert copied is not connector
    assert copied._transport is not transport
    assert copied._transport["headers"] is not transport["headers"]
    assert copied._transport["headers"]["Environment-Name"] == "browser"
    assert copied._transport["headers"]["Environment-Id"] != "env-1"
    assert copied.config is not connector.config
    assert copied.config.include == ["tool1"]
    assert copied.config.exclude == ["tool2"]

    copied._transport["headers"]["Environment-Id"] = "env-3"
    assert copied.config.include is not None
    copied.config.include.append("tool3")

    assert transport["headers"]["Environment-Id"] == "env-1"
    assert connector.config.include == ["tool1"]


def test_eval_context_assigns_unique_hud_environment_ids() -> None:
    """Each copied HUD connector should get a unique environment header set."""
    parent = Environment("parent-env")
    parent_headers = {
        "Environment-Name": "browser",
        "Environment-Id": "parent-env-id",
        "mcp-session-id": "parent-session-id",
    }
    parent._connections["hud"] = Connector(
        transport=SimpleNamespace(url="https://mcp.hud.so/jsonrpc", headers=parent_headers),
        config=ConnectionConfig(),
        name="hud",
        connection_type=ConnectionType.REMOTE,
    )

    ctx_a = EvalContext.from_environment(parent, name="task-a", trace_id="trace-a")
    ctx_b = EvalContext.from_environment(parent, name="task-b", trace_id="trace-b")

    headers_a = ctx_a._connections["hud"]._transport.headers
    headers_b = ctx_b._connections["hud"]._transport.headers

    assert headers_a["Environment-Name"] == "browser"
    assert headers_b["Environment-Name"] == "browser"
    assert headers_a["Environment-Id"] != "parent-env-id"
    assert headers_b["Environment-Id"] != "parent-env-id"
    assert headers_a["Environment-Id"] != headers_b["Environment-Id"]
    assert headers_a is not headers_b
    assert parent_headers["Environment-Id"] == "parent-env-id"
    assert parent_headers["mcp-session-id"] == "parent-session-id"
