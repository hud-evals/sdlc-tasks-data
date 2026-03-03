"""Behavioral tests for CLI release hotfix: version check + gateway validation + force check."""

import os
import sys
from unittest.mock import MagicMock, patch


class TestVersionCheckVirtualenv:
    """Verify that the upgrade command adapts to the execution environment."""

    def _capture_upgrade_message(self, prefix, base_prefix):
        from hud.cli.utils.version_check import VersionInfo, display_update_prompt

        fake = VersionInfo(
            latest="99.0.0", current="0.1.0", is_outdated=True, checked_at=0.0
        )
        console = MagicMock()

        with (
            patch(
                "hud.cli.utils.version_check.check_for_updates", return_value=fake
            ),
            patch.object(sys, "prefix", prefix),
            patch.object(sys, "base_prefix", base_prefix),
        ):
            display_update_prompt(console=console)

        assert console.info.called, "Update prompt was not displayed"
        return console.info.call_args[0][0]

    def test_virtualenv_uses_sync_command(self):
        msg = self._capture_upgrade_message("/home/user/project/.venv", "/usr")
        assert "uv tool upgrade" not in msg, (
            "Virtualenv should NOT suggest 'uv tool upgrade'"
        )
        _VALID_VENV_CMDS = [
            "uv sync --upgrade-package",
            "pip install --upgrade",
            "uv pip install --upgrade",
        ]
        assert any(cmd in msg for cmd in _VALID_VENV_CMDS), (
            f"Expected a virtualenv-appropriate upgrade command, got: {msg}"
        )

    def test_tool_install_uses_tool_command(self):
        msg = self._capture_upgrade_message(
            "/home/user/.local/share/uv/tools/hud-python",
            "/home/user/.local/share/uv/tools/hud-python",
        )
        assert "uv tool upgrade" in msg

    def test_windows_tool_path_uses_tool_command(self):
        msg = self._capture_upgrade_message(
            r"C:\Users\dev\AppData\Local\uv\tools\hud",
            r"C:\Users\dev\AppData\Local\uv\tools\hud",
        )
        assert "uv tool upgrade" in msg

    def test_virtual_env_var_uses_sync_command(self):
        with patch.dict(os.environ, {"VIRTUAL_ENV": "/tmp/project/.venv"}, clear=False):
            msg = self._capture_upgrade_message("/usr/local", "/usr/local")
        assert "uv tool upgrade" not in msg, (
            "VIRTUAL_ENV set — should NOT suggest 'uv tool upgrade'"
        )
        _VALID_VENV_CMDS = [
            "uv sync --upgrade-package",
            "pip install --upgrade",
            "uv pip install --upgrade",
        ]
        assert any(cmd in msg for cmd in _VALID_VENV_CMDS), (
            f"Expected a virtualenv-appropriate upgrade command, got: {msg}"
        )


def _fake_parent_init(self, params=None, **kwargs):
    """Stand-in for MCPAgent.__init__ — sets the attributes child inits expect."""
    self.config = MagicMock()
    self.config.model_client = None
    self.config.validate_api_key = True
    self.console = MagicMock()
    self.hud_console = MagicMock()
    self.system_prompt = ""
    self._available_tools = []


class TestGatewayApiKeyValidation:
    """Verify that validate_api_key only runs with direct provider keys."""

    def test_gemini_gateway_skips_validation(self):
        from hud.agents.gemini import GeminiAgent

        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_settings = MagicMock(api_key="hud-key", gemini_api_key=None)

        with (
            patch("hud.agents.gemini.settings", mock_settings),
            patch("hud.agents.gemini.MCPAgent.__init__", _fake_parent_init),
            patch(
                "hud.agents.gateway.build_gateway_client",
                return_value=mock_client,
            ),
        ):
            GeminiAgent()
            mock_client.models.list.assert_not_called()

    def test_gemini_direct_validates(self):
        from hud.agents.gemini import GeminiAgent

        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_settings = MagicMock(api_key=None, gemini_api_key="real-key")

        with (
            patch("hud.agents.gemini.settings", mock_settings),
            patch("hud.agents.gemini.MCPAgent.__init__", _fake_parent_init),
            patch("hud.agents.gemini.genai.Client", return_value=mock_client),
        ):
            GeminiAgent()
            mock_client.models.list.assert_called()

    def test_openai_gateway_skips_validation(self):
        from hud.agents.openai import OpenAIAgent

        mock_client = MagicMock()
        mock_sync_openai = MagicMock()
        mock_settings = MagicMock(api_key="hud-key", openai_api_key=None)

        with (
            patch("hud.agents.openai.settings", mock_settings),
            patch("hud.agents.openai.MCPAgent.__init__", _fake_parent_init),
            patch(
                "hud.agents.gateway.build_gateway_client",
                return_value=mock_client,
            ),
            patch("hud.agents.openai.OpenAI", mock_sync_openai),
        ):
            OpenAIAgent()
            mock_sync_openai.return_value.models.list.assert_not_called()

    def test_openai_direct_validates(self):
        from hud.agents.openai import OpenAIAgent

        mock_async = MagicMock()
        mock_sync = MagicMock()
        mock_settings = MagicMock(api_key=None, openai_api_key="real-key")

        with (
            patch("hud.agents.openai.settings", mock_settings),
            patch("hud.agents.openai.MCPAgent.__init__", _fake_parent_init),
            patch("hud.agents.openai.AsyncOpenAI", return_value=mock_async),
            patch("hud.agents.openai.OpenAI", return_value=mock_sync),
        ):
            OpenAIAgent()
            mock_sync.models.list.assert_called()


class TestForceVersionCheck:
    """Verify that force_version_check bypasses CI/testing guards."""

    def test_force_check_works_in_ci(self):
        """force_version_check must return results even when CI=true."""
        from hud.cli.utils.version_check import force_version_check

        with (
            patch.dict(os.environ, {"CI": "true"}, clear=False),
            patch(
                "hud.cli.utils.version_check._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "hud.cli.utils.version_check._get_current_version",
                return_value="0.1.0",
            ),
            patch(
                "hud.cli.utils.version_check.VERSION_CACHE_FILE",
                MagicMock(exists=MagicMock(return_value=False)),
            ),
        ):
            result = force_version_check()

        assert result is not None, (
            "force_version_check should return a result even in CI environments"
        )
        assert result.latest == "99.0.0"

    def test_force_check_works_with_skip_flag(self):
        """force_version_check must work even with HUD_SKIP_VERSION_CHECK=1."""
        from hud.cli.utils.version_check import force_version_check

        with (
            patch.dict(
                os.environ, {"HUD_SKIP_VERSION_CHECK": "1"}, clear=False
            ),
            patch(
                "hud.cli.utils.version_check._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "hud.cli.utils.version_check._get_current_version",
                return_value="0.1.0",
            ),
            patch(
                "hud.cli.utils.version_check.VERSION_CACHE_FILE",
                MagicMock(exists=MagicMock(return_value=False)),
            ),
        ):
            result = force_version_check()

        assert result is not None, (
            "force_version_check should return a result even with skip flag"
        )
        assert result.latest == "99.0.0"
