"""Behavioral integration tests for agent configuration and tool compatibility.

Tests verify runtime behavior: imports succeed without circular dependencies,
config classes are the canonical types.py versions (not inline shadows),
and tool categorization uses the shared base class implementation.

These tests do NOT inspect source code or AST — they check the actual Python
objects at runtime, so any correct solution structure will pass.
"""
from __future__ import annotations

import importlib
import sys


def _clear_agent_modules():
    """Clear cached agent modules so each test verifies a fresh import."""
    for key in list(sys.modules):
        if key.startswith("hud.agents"):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# 1. Import health — no circular imports
# ---------------------------------------------------------------------------


class TestImportHealth:
    """Importing agent modules must not trigger circular import errors."""

    def test_import_configs_from_types(self):
        """All config classes must be importable from hud.agents.types."""
        _clear_agent_modules()
        from hud.agents.types import (
            ClaudeConfig,
            GeminiConfig,
            GeminiCUAConfig,
            OpenAIConfig,
            OperatorConfig,
        )
        for cls in (ClaudeConfig, GeminiConfig, GeminiCUAConfig, OpenAIConfig, OperatorConfig):
            assert isinstance(cls, type)

    def test_namespace_import(self):
        """'import hud.agents' must not raise due to circular imports."""
        _clear_agent_modules()
        try:
            import hud.agents  # noqa: F401
        except ImportError as exc:
            raise AssertionError(
                f"Importing hud.agents failed: {exc}\n"
                "This may indicate a circular import."
            ) from exc

    def test_import_all_agents(self):
        """All agent classes must be importable from their modules."""
        _clear_agent_modules()
        from hud.agents.claude import ClaudeAgent
        from hud.agents.gemini import GeminiAgent
        from hud.agents.gemini_cua import GeminiCUAAgent
        from hud.agents.openai import OpenAIAgent
        from hud.agents.operator import OperatorAgent

        for cls in (ClaudeAgent, OpenAIAgent, GeminiAgent, GeminiCUAAgent, OperatorAgent):
            assert isinstance(cls, type)


# ---------------------------------------------------------------------------
# 2. Config identity — must come from types.py, no inline shadows
# ---------------------------------------------------------------------------


class TestConfigIdentity:
    """Each agent module's config must be the exact class from types.py,
    not a locally-defined copy that shadows it."""

    def _assert_no_shadow(self, module_path: str, config_name: str):
        """If *module_path* exposes *config_name*, it must be the same
        object as the one in hud.agents.types (imported, not redefined)."""
        import hud.agents.types as types_mod

        canonical = getattr(types_mod, config_name)
        mod = importlib.import_module(module_path)
        local = mod.__dict__.get(config_name)
        if local is not None and isinstance(local, type):
            assert local is canonical, (
                f"{module_path} defines its own {config_name} that shadows "
                f"types.py. Remove the inline class definition and import "
                f"from hud.agents.types instead."
            )

    def test_claude_config_identity(self):
        self._assert_no_shadow("hud.agents.claude", "ClaudeConfig")

    def test_openai_config_identity(self):
        self._assert_no_shadow("hud.agents.openai", "OpenAIConfig")

    def test_gemini_config_identity(self):
        self._assert_no_shadow("hud.agents.gemini", "GeminiConfig")

    def test_gemini_cua_config_identity(self):
        self._assert_no_shadow("hud.agents.gemini_cua", "GeminiCUAConfig")

    def test_operator_config_identity(self):
        self._assert_no_shadow("hud.agents.operator", "OperatorConfig")

    def test_all_configs_are_pydantic_models(self):
        """Config classes must be Pydantic BaseModel subclasses."""
        from hud.agents.types import (
            ClaudeConfig,
            GeminiConfig,
            GeminiCUAConfig,
            OpenAIConfig,
            OperatorConfig,
        )
        for cls in (ClaudeConfig, GeminiConfig, GeminiCUAConfig, OpenAIConfig, OperatorConfig):
            assert hasattr(cls, "model_fields"), (
                f"{cls.__name__} must be a Pydantic BaseModel"
            )


# ---------------------------------------------------------------------------
# 3. Tool categorization — shared base, no per-agent overrides
# ---------------------------------------------------------------------------


class TestToolCategorization:
    """Agents must use the shared base categorize_tools, not per-agent
    _categorize_tools overrides that shadow it."""

    def test_base_defines_categorize(self):
        """MCPAgent must have the shared categorize_tools method."""
        from hud.agents.base import MCPAgent

        assert hasattr(MCPAgent, "categorize_tools"), (
            "MCPAgent must define categorize_tools as the shared implementation"
        )
        assert callable(MCPAgent.categorize_tools)

    def _assert_no_categorize_override(self, module_path: str, class_name: str):
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        assert "_categorize_tools" not in cls.__dict__, (
            f"{class_name} defines its own _categorize_tools — "
            f"remove it so the shared base class implementation is used"
        )

    def test_claude_no_categorize_override(self):
        self._assert_no_categorize_override("hud.agents.claude", "ClaudeAgent")

    def test_openai_no_categorize_override(self):
        self._assert_no_categorize_override("hud.agents.openai", "OpenAIAgent")

    def test_gemini_no_categorize_override(self):
        self._assert_no_categorize_override("hud.agents.gemini", "GeminiAgent")

    def test_operator_no_categorize_override(self):
        self._assert_no_categorize_override("hud.agents.operator", "OperatorAgent")


# ---------------------------------------------------------------------------
# 4. types.py integrity — module must exist and be the canonical source
# ---------------------------------------------------------------------------


class TestTypesModuleIntegrity:
    """hud.agents.types must exist and contain all canonical config classes."""

    def test_types_importable(self):
        """types.py must be importable (not deleted per PR #455's wrong approach)."""
        mod = importlib.import_module("hud.agents.types")
        assert mod is not None

    def test_types_has_all_required_configs(self):
        """types.py must define all agent config classes."""
        mod = importlib.import_module("hud.agents.types")
        for name in ("ClaudeConfig", "GeminiConfig", "OpenAIConfig", "OperatorConfig"):
            cls = getattr(mod, name, None)
            assert cls is not None, f"types.py missing {name}"
            assert isinstance(cls, type), f"{name} in types.py is not a class"
