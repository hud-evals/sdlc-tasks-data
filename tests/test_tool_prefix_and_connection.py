"""Behavioral tests for prefixed tool routing and connection copy isolation.

These tests verify two critical properties:
1. Tools/resources/prompts imported with a prefix have matching internal names
2. Connector.copy() produces fully isolated copies for parallel execution
"""

import pytest


# ---------------------------------------------------------------------------
# Prefixed Tool Routing
# ---------------------------------------------------------------------------


class TestPrefixedToolRouting:
    """Tools/resources/prompts imported with a prefix must have matching names."""

    def test_prefixed_tool_name_matches_key(self):
        """Tool registered with prefix 'github' should have name 'github_list_issues'."""
        from fastmcp import FastMCP

        from hud.server.server import MCPServer

        router = FastMCP("sub-router")

        @router.tool()
        def list_issues() -> list:
            """List GitHub issues."""
            return []

        server = MCPServer("test-server")
        server.include_router(router, prefix="github")

        for key, tool in server._tool_manager._tools.items():
            if "list_issues" in key:
                assert tool.name == key, (
                    f"Tool key '{key}' doesn't match tool.name '{tool.name}'. "
                    "MCP clients will fail when calling the tool by its listed name."
                )
                break
        else:
            pytest.fail("Tool 'github_list_issues' not found in server tools")

    def test_prefixed_resource_name_matches_key(self):
        """Resource registered with prefix should have matching name."""
        from fastmcp import FastMCP

        from hud.server.server import MCPServer

        router = FastMCP("sub-router")

        @router.resource("resource://configs")
        def get_configs() -> str:
            """Return config."""
            return "{}"

        server = MCPServer("test-server")
        server.include_router(router, prefix="github")

        for key, resource in server._resource_manager._resources.items():
            if "configs" in key:
                assert resource.name == key, (
                    f"Resource key '{key}' doesn't match resource.name '{resource.name}'"
                )
                break
        else:
            pytest.fail("Resource with 'configs' not found in server resources")

    def test_prefixed_prompt_name_matches_key(self):
        """Prompt registered with prefix should have matching name."""
        from fastmcp import FastMCP

        from hud.server.server import MCPServer

        router = FastMCP("sub-router")

        @router.prompt()
        def review_code(code: str) -> str:
            """Review code."""
            return f"Review: {code}"

        server = MCPServer("test-server")
        server.include_router(router, prefix="github")

        for key, prompt in server._prompt_manager._prompts.items():
            if "review_code" in key:
                assert prompt.name == key, (
                    f"Prompt key '{key}' doesn't match prompt.name '{prompt.name}'"
                )
                break
        else:
            pytest.fail("Prompt with 'review_code' not found in server prompts")

    def test_unprefixed_tools_unchanged(self):
        """Tools imported without prefix should keep their original name."""
        from fastmcp import FastMCP

        from hud.server.server import MCPServer

        router = FastMCP("sub-router")

        @router.tool()
        def search() -> list:
            """Search."""
            return []

        server = MCPServer("test-server")
        server.include_router(router)  # No prefix

        for key, tool in server._tool_manager._tools.items():
            if key == "search":
                assert tool.name == "search"
                break
        else:
            pytest.fail("Tool 'search' not found")


# ---------------------------------------------------------------------------
# Connector Copy Isolation
# ---------------------------------------------------------------------------


class TestConnectorCopyIsolation:
    """Connector.copy() must produce fully isolated copies for parallel execution."""

    def test_copy_transport_is_independent(self):
        """Mutating copied transport must not affect original."""
        from hud.environment.connection import (
            ConnectionConfig,
            ConnectionType,
            Connector,
        )

        transport = {
            "url": "https://mcp.hud.ai/jsonrpc",
            "headers": {
                "Environment-Name": "browser",
                "Environment-Id": "original-env-id",
            },
        }
        connector = Connector(
            transport=transport,
            config=ConnectionConfig(),
            name="hud",
            connection_type=ConnectionType.REMOTE,
        )

        copied = connector.copy()

        # Mutate the copy's transport headers
        copied._transport["headers"]["Environment-Id"] = "mutated-by-copy"

        # Original must be unaffected
        assert transport["headers"]["Environment-Id"] == "original-env-id", (
            "Connector.copy() shares transport reference — "
            "parallel traces will stomp each other's session state"
        )

    def test_copy_config_is_independent(self):
        """Mutating copied config must not affect original."""
        from hud.environment.connection import (
            ConnectionConfig,
            ConnectionType,
            Connector,
        )

        config = ConnectionConfig(include=["tool1"], exclude=["tool2"])
        connector = Connector(
            transport={"url": "https://example.com"},
            config=config,
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        copied = connector.copy()

        # Mutate the copy's config lists
        assert copied.config.include is not None
        copied.config.include.append("tool3")

        # Original must be unaffected
        assert config.include == ["tool1"], (
            "Connector.copy() shares config reference — "
            "modifications leak between parallel traces"
        )

    def test_copy_hud_server_gets_new_environment_id(self):
        """Copy of HUD server connector must get a fresh Environment-Id."""
        from hud.environment.connection import (
            ConnectionConfig,
            ConnectionType,
            Connector,
        )

        transport = {
            "url": "https://mcp.hud.ai/jsonrpc",
            "headers": {
                "Environment-Name": "browser",
                "Environment-Id": "parent-env-id",
            },
        }
        connector = Connector(
            transport=transport,
            config=ConnectionConfig(),
            name="hud",
            connection_type=ConnectionType.REMOTE,
        )

        copied = connector.copy()

        copied_id = copied._transport.get("headers", {}).get("Environment-Id")
        assert copied_id != "parent-env-id", (
            "HUD server copy must generate a new Environment-Id "
            "to prevent parallel traces from sharing sessions"
        )

    def test_copy_non_hud_preserves_headers(self):
        """Copy of non-HUD connector preserves header values (in separate objects)."""
        from hud.environment.connection import (
            ConnectionConfig,
            ConnectionType,
            Connector,
        )

        transport = {
            "url": "https://example.com/mcp",
            "headers": {
                "Authorization": "Bearer token-123",
            },
        }
        connector = Connector(
            transport=transport,
            config=ConnectionConfig(),
            name="external",
            connection_type=ConnectionType.REMOTE,
        )

        copied = connector.copy()

        # Non-HUD servers should preserve header values
        copied_headers = copied._transport.get("headers", {})
        assert copied_headers.get("Authorization") == "Bearer token-123"

        # But transport must still be independent
        copied._transport["headers"]["Authorization"] = "Bearer CHANGED"
        assert transport["headers"]["Authorization"] == "Bearer token-123", (
            "Even non-HUD connector copies must have independent transport"
        )
