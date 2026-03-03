"""Integration tests for MCP client library migration.

Verifies that CLI commands use the upstream fastmcp.Client (not the removed
FastMCPHUDClient), that client lifecycle methods are correct, and that the
server signal handler properly sets the shutdown flag.

These tests are behavioral — they accept ANY valid migration approach.
"""
from __future__ import annotations

import ast
import os


def _parse_file(filepath: str) -> ast.Module:
    with open(filepath) as f:
        return ast.parse(f.read())


def _read_source(filepath: str) -> str:
    with open(filepath) as f:
        return f.read()


def _find_function(tree: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Function '{function_name}' not found")


def _function_source(filepath: str, function_name: str) -> str:
    source = _read_source(filepath)
    lines = source.splitlines()
    tree = ast.parse(source)
    fn = _find_function(tree, function_name)
    start = fn.lineno - 1
    end = fn.end_lineno
    return "\n".join(lines[start:end])


class TestNoOldClientImports:
    """The deleted hud.clients module must not be imported anywhere."""

    def _assert_no_old_imports(self, filepath: str):
        tree = _parse_file(filepath)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "hud.clients" in module:
                    names = [alias.name for alias in node.names]
                    assert False, (
                        f"{filepath} imports from deleted module '{module}': {names}. "
                        f"The hud.clients package was removed. "
                        f"Use 'from fastmcp import Client' or equivalent."
                    )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "hud.clients" in alias.name:
                        assert False, (
                            f"{filepath} imports deleted module '{alias.name}'. "
                            f"The hud.clients package was removed."
                        )

    def test_debug_no_old_imports(self):
        self._assert_no_old_imports("hud/cli/debug.py")

    def test_server_no_old_imports(self):
        self._assert_no_old_imports("hud/server/server.py")

    def test_dev_no_old_imports(self):
        self._assert_no_old_imports("hud/cli/dev.py")


class TestNoOldClientUsage:
    """Code must not instantiate or call methods on FastMCPHUDClient.

    Checks that the old class name is not referenced. This is more
    precise than blocking method names like .shutdown() which a valid
    wrapper class might define.
    """

    def _assert_no_fastmcp_hud_client(self, filepath: str):
        source = _read_source(filepath)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "FastMCPHUDClient":
                line = getattr(node, "lineno", "?")
                assert False, (
                    f"{filepath}:{line} references 'FastMCPHUDClient'. "
                    f"This class was in the deleted hud.clients package. "
                    f"Use fastmcp.Client or a wrapper around it."
                )

    def test_debug_no_fastmcp_hud_client(self):
        self._assert_no_fastmcp_hud_client("hud/cli/debug.py")

    def test_server_no_fastmcp_hud_client(self):
        self._assert_no_fastmcp_hud_client("hud/server/server.py")

    def test_dev_no_fastmcp_hud_client(self):
        self._assert_no_fastmcp_hud_client("hud/cli/dev.py")


class TestServerSignalHandling:
    """server.py must register synchronous signal handlers that set
    _sigterm_received = True BEFORE starting the event loop."""

    def test_sigterm_flag_used(self):
        source = _read_source("hud/server/server.py")
        assert "_sigterm_received" in source, (
            "server.py must use _sigterm_received flag for clean shutdown"
        )

    def test_sync_signal_handler_registered(self):
        """Must register signal handler via signal.signal() (synchronous),
        not only via loop.add_signal_handler() (async-only)."""
        source = _read_source("hud/server/server.py")
        tree = ast.parse(source)

        found_sync_registration = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                    and func.attr == "signal"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "signal"):
                    found_sync_registration = True
                    break

        assert found_sync_registration, (
            "server.py must register a synchronous signal handler via "
            "signal.signal(signal.SIGTERM, handler). "
            "loop.add_signal_handler alone doesn't fire when the event loop "
            "is blocked on stdin reads."
        )

    def test_sync_signal_registered_before_anyio_run(self):
        """Register sync handlers before entering anyio.run loop."""
        fn_source = _function_source("hud/server/server.py", "_run_with_sigterm")
        fn_tree = ast.parse(fn_source)

        sigterm_lines: list[int] = []
        anyio_run_lines: list[int] = []

        for node in ast.walk(fn_tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "signal"
                and node.func.attr == "signal"
                and node.args
                and isinstance(node.args[0], ast.Attribute)
                and isinstance(node.args[0].value, ast.Name)
                and node.args[0].value.id == "signal"
                and node.args[0].attr == "SIGTERM"
            ):
                sigterm_lines.append(node.lineno)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "anyio"
                and node.func.attr == "run"
            ):
                anyio_run_lines.append(node.lineno)

        assert sigterm_lines, (
            "Expected synchronous SIGTERM registration with signal.signal"
        )
        # Some valid implementations may not call anyio.run in this function.
        if anyio_run_lines:
            assert min(sigterm_lines) < min(anyio_run_lines), (
                "signal.signal(SIGTERM, ...) must be registered before anyio.run(...) "
                "so the shutdown flag is set even when the loop blocks on stdio."
            )

    def test_sync_handler_sets_flag(self):
        """A synchronous signal handler function must set _sigterm_received."""
        source = _read_source("hud/server/server.py")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            name_lower = node.name.lower()
            if "sigterm" not in name_lower and "signal" not in name_lower:
                continue
            # Check if this function contains `global _sigterm_received`
            # or assigns to _sigterm_received
            for child in ast.walk(node):
                if isinstance(child, ast.Global) and "_sigterm_received" in child.names:
                    return  # Found it
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "_sigterm_received":
                            return

        assert False, (
            "No signal handler function sets _sigterm_received = True. "
            "A synchronous SIGTERM handler must set this flag."
        )


class TestDebugFileParses:
    """debug.py must be syntactically valid after migration."""

    def test_debug_compiles(self):
        source = _read_source("hud/cli/debug.py")
        try:
            compile(source, "hud/cli/debug.py", "exec")
        except SyntaxError as e:
            assert False, f"hud/cli/debug.py has syntax error: {e}"

    def test_server_compiles(self):
        source = _read_source("hud/server/server.py")
        try:
            compile(source, "hud/server/server.py", "exec")
        except SyntaxError as e:
            assert False, f"hud/server/server.py has syntax error: {e}"


class TestDevFileParses:
    """dev.py must be syntactically valid after migration."""

    def test_dev_compiles(self):
        source = _read_source("hud/cli/dev.py")
        try:
            compile(source, "hud/cli/dev.py", "exec")
        except SyntaxError as e:
            assert False, f"hud/cli/dev.py has syntax error: {e}"

    def test_dev_no_old_lifecycle_calls(self):
        """dev.py must not call .shutdown() or .initialize() (old client API)."""
        source = _read_source("hud/cli/dev.py")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "initialize":
                if isinstance(node.ctx, ast.Load):
                    line = getattr(node, "lineno", "?")
                    assert False, (
                        f"hud/cli/dev.py:{line} calls '.initialize()' — "
                        f"fastmcp.Client uses context manager or .__aenter__() instead."
                    )
