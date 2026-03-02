"""Integration tests for MCP connector routing.

Verifies that connect_hub passes include/exclude filters correctly,
and that _add_connection uses the prefix parameter (not name) for
tool namespacing.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestConnectHubFiltering:
    """connect_hub must forward include/exclude filters."""

    def test_connect_hub_forwards_include(self):
        """connect_hub's include param must reach connect_mcp_config."""
        from hud.environment.connectors.remote import RemoteConnectorMixin

        mock_self = MagicMock()
        mock_self._hub_config = None
        mock_self.connect_mcp_config = MagicMock()

        mock_settings = MagicMock()
        mock_settings.hud_mcp_url = "https://mcp.test/v3/mcp"

        with patch("hud.settings.settings", mock_settings):
            RemoteConnectorMixin.connect_hub(
                mock_self,
                "test-slug",
                include=["bash", "edit"],
                prefix="coding",
            )

        mock_self.connect_mcp_config.assert_called_once()
        _, kwargs = mock_self.connect_mcp_config.call_args
        assert kwargs.get("include") == ["bash", "edit"], (
            f"connect_hub() did not forward include=['bash', 'edit'] "
            f"to connect_mcp_config(). Got: {kwargs}"
        )

    def test_connect_hub_forwards_exclude(self):
        """connect_hub's exclude param must reach connect_mcp_config."""
        from hud.environment.connectors.remote import RemoteConnectorMixin

        mock_self = MagicMock()
        mock_self._hub_config = None
        mock_self.connect_mcp_config = MagicMock()

        mock_settings = MagicMock()
        mock_settings.hud_mcp_url = "https://mcp.test/v3/mcp"

        with patch("hud.settings.settings", mock_settings):
            RemoteConnectorMixin.connect_hub(
                mock_self,
                "test-slug",
                exclude=["admin_debug"],
            )

        mock_self.connect_mcp_config.assert_called_once()
        _, kwargs = mock_self.connect_mcp_config.call_args
        assert kwargs.get("exclude") == ["admin_debug"], (
            f"connect_hub() did not forward exclude=['admin_debug'] "
            f"to connect_mcp_config(). Got: {kwargs}"
        )

    def test_connect_hub_forwards_both_filters(self):
        """connect_hub with both include and exclude forwards both."""
        from hud.environment.connectors.remote import RemoteConnectorMixin

        mock_self = MagicMock()
        mock_self._hub_config = None
        mock_self.connect_mcp_config = MagicMock()

        mock_settings = MagicMock()
        mock_settings.hud_mcp_url = "https://mcp.test/v3/mcp"

        with patch("hud.settings.settings", mock_settings):
            RemoteConnectorMixin.connect_hub(
                mock_self,
                "test-slug",
                include=["bash"],
                exclude=["admin"],
                prefix="hub",
            )

        _, kwargs = mock_self.connect_mcp_config.call_args
        assert kwargs.get("include") == ["bash"], "include not forwarded"
        assert kwargs.get("exclude") == ["admin"], "exclude not forwarded"


class TestPrefixNamespacing:
    """_add_connection must use prefix parameter for tool namespacing."""

    def test_prefix_used_not_name(self):
        """ConnectionConfig must receive the prefix param, not the name."""
        from hud.environment.connectors.base import BaseConnectorMixin
        from hud.environment.connection import ConnectionConfig

        captured = []
        real_init = ConnectionConfig.__init__

        def spy_init(self_cfg, **kwargs):
            captured.append(kwargs)
            real_init(self_cfg, **kwargs)

        mock_self = MagicMock()
        mock_self._connections = {}

        with patch.object(ConnectionConfig, "__init__", spy_init):
            with patch("hud.environment.connection.Connector"):
                try:
                    BaseConnectorMixin._add_connection(
                        mock_self,
                        "server-name",
                        MagicMock(),
                        connection_type=MagicMock(),
                        prefix="hub-prefix",
                    )
                except Exception:
                    pass

        assert len(captured) >= 1, "ConnectionConfig was never instantiated"
        assert captured[0].get("prefix") == "hub-prefix", (
            f"ConnectionConfig(prefix='{captured[0].get('prefix')}') — "
            f"expected 'hub-prefix' but got the connection name instead. "
            f"_add_connection must pass prefix=prefix, not prefix=name."
        )

    def test_connection_config_has_prefix_field(self):
        """ConnectionConfig must have a prefix field."""
        from hud.environment.connection import ConnectionConfig

        config = ConnectionConfig(prefix="hub", include=None, exclude=None)
        assert config.prefix == "hub", (
            f"ConnectionConfig.prefix is '{config.prefix}', expected 'hub'"
        )

    def test_prefix_none_by_default(self):
        """ConnectionConfig prefix defaults to None (no namespacing)."""
        from hud.environment.connection import ConnectionConfig

        config = ConnectionConfig()
        assert config.prefix is None, (
            f"ConnectionConfig.prefix defaults to '{config.prefix}', expected None"
        )

    def test_prefix_not_equal_to_name(self):
        """When prefix differs from name, ConnectionConfig must use prefix."""
        from hud.environment.connectors.base import BaseConnectorMixin
        from hud.environment.connection import ConnectionConfig

        captured = []
        real_init = ConnectionConfig.__init__

        def spy_init(self_cfg, **kwargs):
            captured.append(kwargs)
            real_init(self_cfg, **kwargs)

        mock_self = MagicMock()
        mock_self._connections = {}

        with patch.object(ConnectionConfig, "__init__", spy_init):
            with patch("hud.environment.connection.Connector"):
                try:
                    BaseConnectorMixin._add_connection(
                        mock_self,
                        "my-server",
                        MagicMock(),
                        connection_type=MagicMock(),
                        prefix="custom-ns",
                    )
                except Exception:
                    pass

        assert len(captured) >= 1
        prefix_val = captured[0].get("prefix")
        assert prefix_val != "my-server", (
            f"ConnectionConfig(prefix='{prefix_val}') is using the connection "
            f"name instead of the prefix parameter."
        )
        assert prefix_val == "custom-ns", (
            f"ConnectionConfig(prefix='{prefix_val}'), expected 'custom-ns'"
        )
