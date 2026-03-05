"""Behavioral tests for folder-mode .env propagation across CLI paths.

Tests verify that CLI commands (debug, dev, build) correctly propagate .env
variables to Docker commands in folder mode, without prescribing specific
function names or implementation patterns.

Strategy:
- Build path: mock analyze_mcp_environment, call build_environment, verify
  env_vars include .env values. Works regardless of helper function names.
- Debug path: mock debug_mcp_stdio, call the debug function, verify docker
  command includes -e flags from .env.
- Dev path: verify ad-hoc .env parsing replaced by shared logic.
- Image-only: verify build_run_command unchanged.
"""

import importlib
import inspect
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _get_build_module():
    """Get the hud.cli.build MODULE (not the build() typer command function).

    hud.cli.__init__ defines a build() function that shadows the submodule
    in attribute lookup. We use sys.modules to get the actual module.
    """
    importlib.import_module("hud.cli.build")
    return sys.modules["hud.cli.build"]


def _write_env_file(directory: Path, content: str) -> Path:
    """Write a .env file into the given directory and return the directory."""
    env_file = directory / ".env"
    env_file.write_text(textwrap.dedent(content).strip() + "\n")
    return directory


def _setup_env_dir(tmp_path: Path, env_content: str | None = None) -> Path:
    """Create a minimal environment directory with optional .env."""
    if env_content is not None:
        _write_env_file(tmp_path, env_content)
    (tmp_path / "Dockerfile.hud").write_text("FROM python:3.12\n")
    return tmp_path


class _AnalysisCaptured(Exception):
    """Raised by mock to stop build_environment after capturing env_vars."""


def _run_build_capture_analysis_env(
    tmp_path: Path,
    monkeypatch,
    env_content: str | None = None,
    explicit_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Run build_environment with mocks, return env_vars received by analysis.

    The mock analyze_mcp_environment captures the env_vars and then raises
    to stop execution cleanly — avoiding the need to mock all downstream
    code (lock-file writes, runtime metadata, etc.).
    """
    _setup_env_dir(tmp_path, env_content)

    captured: dict[str, dict] = {}

    async def mock_analyze(image, verbose=False, env_vars=None):
        captured["env_vars"] = dict(env_vars) if env_vars else {}
        raise _AnalysisCaptured("captured")

    build_mod = _get_build_module()

    monkeypatch.setattr(build_mod, "analyze_mcp_environment", mock_analyze)
    monkeypatch.setattr(build_mod, "build_docker_image", lambda *a, **kw: True)
    monkeypatch.setattr(build_mod, "HUDConsole", lambda: MagicMock())

    import hud.cli.utils.docker as docker_mod

    monkeypatch.setattr(docker_mod, "require_docker_running", lambda: None)

    try:
        build_mod.build_environment(
            directory=str(tmp_path), env_vars=explicit_env
        )
    except BaseException:
        pass

    return captured.get("env_vars", {})


class TestBuildAnalysisEnvPropagation:
    """build_environment must merge .env into env_vars for analysis."""

    def test_dotenv_values_reach_analysis(self, tmp_path, monkeypatch):
        """analyze_mcp_environment must receive values from .env."""
        env = _run_build_capture_analysis_env(
            tmp_path, monkeypatch, env_content="FROM_FILE=yes\nTEST_VAR=123"
        )
        assert env.get("FROM_FILE") == "yes", f"Missing FROM_FILE: {env}"
        assert env.get("TEST_VAR") == "123", f"Missing TEST_VAR: {env}"

    def test_explicit_env_overrides_dotenv(self, tmp_path, monkeypatch):
        """Explicit env_vars must override same-key values from .env."""
        env = _run_build_capture_analysis_env(
            tmp_path,
            monkeypatch,
            env_content="API_KEY=from_file\nKEEP=yes",
            explicit_env={"API_KEY": "from_caller"},
        )
        assert env.get("API_KEY") == "from_caller", f"Override failed: {env}"
        assert env.get("KEEP") == "yes", f"Non-overridden key lost: {env}"

    def test_no_dotenv_passes_explicit_only(self, tmp_path, monkeypatch):
        """Without .env, only explicit env_vars should reach analysis."""
        env = _run_build_capture_analysis_env(
            tmp_path, monkeypatch, env_content=None, explicit_env={"ONLY": "val"}
        )
        assert env.get("ONLY") == "val"


class TestDebugFolderMode:
    """Debug folder-mode must produce docker command with .env flags."""

    def test_folder_mode_command_includes_env(self, tmp_path, monkeypatch):
        """Docker command passed to debug_mcp_stdio must include -e flags
        from the folder's .env file."""
        _write_env_file(tmp_path, "INJECTED_VAR=from_dotenv\nSECOND=val2")
        (tmp_path / "Dockerfile.hud").write_text("FROM python:3.12\n")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "0.1.0"\n'
        )

        captured_command: list[str] = []

        async def mock_debug_stdio(command, logger, max_phase=None):
            captured_command.extend(command)
            return 3

        import hud.cli as cli_mod
        import hud.cli.utils.environment as env_mod

        monkeypatch.setattr(cli_mod, "debug_mcp_stdio", mock_debug_stdio)
        monkeypatch.setattr(cli_mod, "CaptureLogger", lambda **kw: MagicMock())
        monkeypatch.setattr(cli_mod, "HUDConsole", lambda: MagicMock())
        monkeypatch.setattr(env_mod, "get_image_name", lambda x: ("test:img", "auto"))
        monkeypatch.setattr(env_mod, "image_exists", lambda x: True)
        monkeypatch.setattr(env_mod, "build_environment", lambda *a, **kw: True)

        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli_mod.app, ["debug", str(tmp_path)])

        joined = " ".join(str(x) for x in captured_command)
        assert "INJECTED_VAR=from_dotenv" in joined, (
            f"Expected INJECTED_VAR in debug docker command: {joined}"
        )
        assert "SECOND=val2" in joined, (
            f"Expected SECOND in debug docker command: {joined}"
        )


class TestDevEnvConsistency:
    """Dev mode must use centralized env loading, not ad-hoc inline parsing."""

    def test_no_adhoc_inline_env_parsing(self):
        """run_docker_dev_server must not contain ad-hoc .env parsing.

        The baseline reads .env line-by-line and extends docker_cmd inline;
        after the fix this should be replaced by shared helpers.
        """
        from hud.cli.dev import run_docker_dev_server

        source = inspect.getsource(run_docker_dev_server)

        adhoc_markers = [
            'env_file.read_text().splitlines()',
            'docker_cmd.extend(["-e", line])',
        ]
        for marker in adhoc_markers:
            assert marker not in source, (
                f"run_docker_dev_server still contains ad-hoc .env parsing: "
                f"found '{marker}'"
            )


class TestImageOnlyRegression:
    """Image-only mode must remain unchanged — no implicit .env loading."""

    def test_build_run_command_no_env_injection(self):
        """build_run_command (image-only path) must not inject .env flags."""
        from hud.cli.utils.docker import build_run_command

        cmd = build_run_command("registry/my-image:v1", docker_args=["-p", "3000:3000"])
        assert cmd[0] == "docker"
        assert cmd[-1] == "registry/my-image:v1"
        e_values = [
            cmd[i + 1]
            for i in range(len(cmd))
            if cmd[i] == "-e" and i + 1 < len(cmd)
        ]
        assert len(e_values) == 0, f"Image-only mode leaked env: {e_values}"

    def test_build_run_command_preserves_args(self):
        """build_run_command must include all user docker_args."""
        from hud.cli.utils.docker import build_run_command

        cmd = build_run_command("img:tag", docker_args=["-v", "/a:/b", "-p", "80:80"])
        joined = " ".join(cmd)
        assert "/a:/b" in joined
        assert "80:80" in joined
        assert cmd[-1] == "img:tag"


class TestEnvFileEdgeCases:
    """Edge cases in .env file handling must not crash or inject bad values."""

    def test_empty_env_file_no_crash(self, tmp_path, monkeypatch):
        """An empty .env file must not cause errors."""
        _run_build_capture_analysis_env(tmp_path, monkeypatch, env_content="")

    def test_comments_and_blank_lines_ignored(self, tmp_path, monkeypatch):
        """Comments and blank lines in .env must be ignored."""
        env = _run_build_capture_analysis_env(
            tmp_path,
            monkeypatch,
            env_content="# This is a comment\n\nREAL_KEY=real_val\n# Another\n",
        )
        assert env.get("REAL_KEY") == "real_val", f"REAL_KEY missing: {env}"
        assert "# This is a comment" not in str(env.values())
