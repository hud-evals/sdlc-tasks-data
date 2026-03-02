"""Integration tests for task loading and model resolution.

Verifies that JSONL loading handles multi-line prompts, v4 task conversion
produces independent mcp_config, gateway model resolution retries after
initial failure, and Gemini client uses correct API version and auth format.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch


class TestJsonlMultilineLoading:
    """JSONL parser must handle multi-line prompt strings."""

    def test_multiline_prompt_loaded(self):
        """All tasks loaded even when prompts contain escaped newlines."""
        from hud.datasets.loader import _load_raw_from_file
        from pathlib import Path

        tasks = [
            {"prompt": "Line 1\nLine 2\nLine 3", "mcp_config": {}, "evaluate_tool": []},
            {"prompt": "Simple prompt", "mcp_config": {}, "evaluate_tool": []},
            {"prompt": "Code:\n```python\nprint('hello')\n```", "mcp_config": {}, "evaluate_tool": []},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
            path = f.name

        try:
            result = _load_raw_from_file(Path(path))
            assert len(result) == 3, f"Expected 3 tasks, got {len(result)}"
        finally:
            os.unlink(path)

    def test_prompt_preserved_exactly(self):
        """Multi-line prompt content is preserved without data loss."""
        from hud.datasets.loader import _load_raw_from_file
        from pathlib import Path

        original_prompt = "Step 1: Navigate to the page\nStep 2: Click the button\nStep 3: Verify"
        tasks = [{"prompt": original_prompt, "mcp_config": {}, "evaluate_tool": []}]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
            path = f.name

        try:
            result = _load_raw_from_file(Path(path))
            assert result[0]["prompt"] == original_prompt, (
                f"Prompt was modified during loading: {result[0]['prompt']!r}"
            )
        finally:
            os.unlink(path)

    def test_empty_lines_skipped(self):
        """Blank lines in JSONL don't cause crashes."""
        from hud.datasets.loader import _load_raw_from_file
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "task1"}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"prompt": "task2"}) + "\n")
            path = f.name

        try:
            result = _load_raw_from_file(Path(path))
            assert len(result) == 2, f"Expected 2 tasks (skipping blanks), got {len(result)}"
        finally:
            os.unlink(path)


class TestModelResolverCache:
    """Gateway model resolution must retry after initial failures."""

    def test_retry_after_key_becomes_available(self):
        """Models can be fetched after API key is configured."""
        import hud.agents.resolver as resolver

        old_cache = resolver._models_cache
        try:
            resolver._models_cache = None

            with patch("hud.settings.settings", MagicMock(api_key=None)):
                result1 = resolver._fetch_gateway_models()
                assert result1 == []

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": [{"id": "m1", "name": "Claude"}]}
            mock_resp.raise_for_status = MagicMock()
            with patch("hud.settings.settings", MagicMock(api_key="real-key", hud_api_url="http://api")):
                with patch("httpx.get", return_value=mock_resp) as mock_get:
                    result2 = resolver._fetch_gateway_models()

            mock_get.assert_called_once()
            assert len(result2) == 1, (
                "Should have fetched models after API key became available, "
                "but got cached empty result"
            )
        finally:
            resolver._models_cache = old_cache

    def test_retry_after_network_failure(self):
        """Transient network errors don't prevent future fetches."""
        import hud.agents.resolver as resolver

        old_cache = resolver._models_cache
        try:
            resolver._models_cache = None

            with patch("hud.settings.settings", MagicMock(api_key="key", hud_api_url="http://api")):
                with patch("httpx.get", side_effect=ConnectionError("network")):
                    result1 = resolver._fetch_gateway_models()
                    assert result1 == []

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": [{"id": "m1"}]}
            mock_resp.raise_for_status = MagicMock()
            with patch("hud.settings.settings", MagicMock(api_key="key", hud_api_url="http://api")):
                with patch("httpx.get", return_value=mock_resp) as mock_get:
                    result2 = resolver._fetch_gateway_models()

            mock_get.assert_called_once()
            assert len(result2) == 1, (
                "Should have fetched models after network recovered"
            )
        finally:
            resolver._models_cache = old_cache

    def test_success_is_cached(self):
        """Successful fetch IS cached — no redundant API calls."""
        import hud.agents.resolver as resolver

        old_cache = resolver._models_cache
        try:
            resolver._models_cache = None

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": [{"id": "m1", "name": "Claude"}]}
            mock_resp.raise_for_status = MagicMock()

            with patch("hud.settings.settings", MagicMock(api_key="key", hud_api_url="http://api")):
                with patch("httpx.get", return_value=mock_resp) as mock_get:
                    result1 = resolver._fetch_gateway_models()
                    result2 = resolver._fetch_gateway_models()

            mock_get.assert_called_once()
            assert len(result1) == 1
            assert len(result2) == 1
        finally:
            resolver._models_cache = old_cache


class TestGatewayClientConstruction:
    """Gateway clients must be built correctly per provider."""

    def test_gemini_client_api_version(self):
        """Gemini client must use v1beta API version."""
        mock_settings = MagicMock(api_key="test-key", hud_gateway_url="https://gateway.hud.ai")
        mock_http_options_cls = MagicMock()
        mock_genai = MagicMock()

        with patch("hud.settings.settings", mock_settings):
            with patch.dict("sys.modules", {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": MagicMock(HttpOptions=mock_http_options_cls),
            }):
                from importlib import reload
                import hud.agents.gateway as gw
                reload(gw)
                gw.build_gateway_client("gemini")

                call_args = mock_http_options_cls.call_args
                api_version = call_args[1].get("api_version") if call_args[1] else call_args[0][0]
                assert api_version == "v1beta", (
                    f"Gemini uses api_version='{api_version}', must be 'v1beta'"
                )

    def test_gemini_client_auth_header(self):
        """Gemini auth header must use Bearer prefix."""
        mock_settings = MagicMock(api_key="test-key-123", hud_gateway_url="https://gateway.hud.ai")
        mock_http_options_cls = MagicMock()
        mock_genai = MagicMock()

        with patch("hud.settings.settings", mock_settings):
            with patch.dict("sys.modules", {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": MagicMock(HttpOptions=mock_http_options_cls),
            }):
                from importlib import reload
                import hud.agents.gateway as gw
                reload(gw)
                gw.build_gateway_client("gemini")

                call_args = mock_http_options_cls.call_args
                headers = call_args[1].get("headers", {})
                auth = headers.get("Authorization", "")
                assert auth.startswith("Bearer "), (
                    f"Gemini auth header is '{auth}' — must start with 'Bearer '"
                )
