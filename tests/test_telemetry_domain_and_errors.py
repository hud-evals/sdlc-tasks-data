"""Hidden tests for telemetry domain compatibility.

Verifies that MCP config patching works for all HUD domains (not just the
current default). After the hud.so -> hud.ai migration, users with old configs
must still get auth headers injected.

All tests are BEHAVIORAL — they call patch_mcp_config() with real inputs and
check real outputs. Any valid fix passes.
"""

from __future__ import annotations


class TestPatchMcpConfigDomainCompat:
    """patch_mcp_config must inject headers for both old (.so) and new (.ai) HUD domains."""

    def test_patches_headers_for_hud_ai(self):
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {"server1": {"url": "https://mcp.hud.ai/v3/mcp"}}
        p = MCPConfigPatch(headers={"Authorization": "Bearer test-key"})
        patch_mcp_config(config, p)
        assert config["server1"]["headers"]["Authorization"] == "Bearer test-key"

    def test_patches_headers_for_hud_so(self):
        """Critical: old-domain configs must still get auth headers after migration."""
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {"server1": {"url": "https://mcp.hud.so/v3/mcp"}}
        p = MCPConfigPatch(headers={"Authorization": "Bearer test-key"})
        patch_mcp_config(config, p)
        assert "headers" in config["server1"], (
            "Auth headers not injected for old hud.so domain — this is the root cause "
            "of telemetry failures after the domain migration"
        )
        assert config["server1"]["headers"]["Authorization"] == "Bearer test-key"

    def test_patches_headers_for_any_hud_domain(self):
        """Should work for any future hud.* domain too."""
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {"server1": {"url": "https://mcp.hud.dev/v3/mcp"}}
        p = MCPConfigPatch(headers={"Authorization": "Bearer test-key"})
        patch_mcp_config(config, p)
        assert "headers" in config["server1"]

    def test_does_not_patch_non_hud_server(self):
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {"server1": {"url": "https://mcp.example.com/v3/mcp"}}
        p = MCPConfigPatch(headers={"Authorization": "Bearer test-key"})
        patch_mcp_config(config, p)
        assert "headers" not in config["server1"]

    def test_patches_metadata_for_all_servers(self):
        """Metadata lane should work for any server, not just HUD ones."""
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {"server1": {"url": "https://mcp.example.com/v3/mcp"}}
        p = MCPConfigPatch(meta={"run_id": "test-123"})
        patch_mcp_config(config, p)
        assert config["server1"]["meta"]["run_id"] == "test-123"

    def test_mixed_domains_only_hud_gets_headers(self):
        """In a multi-server config, only HUD servers should get auth headers."""
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {
            "hud_old": {"url": "https://mcp.hud.so/v3/mcp"},
            "hud_new": {"url": "https://mcp.hud.ai/v3/mcp"},
            "external": {"url": "https://mcp.other.com/v3/mcp"},
        }
        p = MCPConfigPatch(headers={"Authorization": "Bearer key"})
        patch_mcp_config(config, p)
        assert "headers" in config["hud_old"], "Old domain should get headers"
        assert "headers" in config["hud_new"], "New domain should get headers"
        assert "headers" not in config["external"], "External should NOT get headers"

    def test_case_insensitive_domain_matching(self):
        """Domain matching should be case-insensitive."""
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {"server1": {"url": "https://MCP.HUD.AI/v3/mcp"}}
        p = MCPConfigPatch(headers={"Authorization": "Bearer test-key"})
        patch_mcp_config(config, p)
        assert "headers" in config["server1"]

    def test_empty_url_does_not_crash(self):
        """Servers with empty URLs should not crash the patching."""
        from hud.utils.mcp import MCPConfigPatch, patch_mcp_config

        config = {"server1": {"url": ""}}
        p = MCPConfigPatch(headers={"Authorization": "Bearer test-key"})
        patch_mcp_config(config, p)
        assert "headers" not in config["server1"]
