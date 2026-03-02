"""Tests for coding tool sandbox security: path traversal and bash sentinel handling."""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.coding.apply_patch import ApplyPatchTool, DiffError
from hud.tools.coding.bash import ClaudeBashSession
from hud.tools.types import ToolError


# ---------------------------------------------------------------------------
# Path traversal prevention
# ---------------------------------------------------------------------------


class TestPathTraversalPrevention:
    """Verify _validate_path blocks sibling directory access via prefix bypass."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tool = ApplyPatchTool.__new__(ApplyPatchTool)
        self.tool.base_path = os.path.abspath(self.tmpdir)

    def test_sibling_directory_blocked(self):
        """A path resolving to a sibling of base_path must be rejected.

        E.g., base_path=/tmp/myapp → /tmp/myapp_sibling/file should fail.
        """
        sibling = self.tool.base_path + "_sibling"
        os.makedirs(sibling, exist_ok=True)
        with open(os.path.join(sibling, "secret.txt"), "w") as f:
            f.write("secret")
        with pytest.raises(DiffError, match="Path traversal"):
            self.tool._validate_path("../" + os.path.basename(sibling) + "/secret.txt")

    def test_exact_base_path_allowed(self):
        """A path resolving to base_path itself should be allowed."""
        result = self.tool._validate_path(".")
        assert result == self.tool.base_path

    def test_subdirectory_allowed(self):
        """A path within base_path should be allowed."""
        subdir = os.path.join(self.tmpdir, "src")
        os.makedirs(subdir, exist_ok=True)
        result = self.tool._validate_path("src")
        assert result == subdir

    def test_parent_traversal_blocked(self):
        """Absolute-style traversal (../../etc/passwd) must be rejected."""
        with pytest.raises(DiffError):
            self.tool._validate_path("../../../etc/passwd")


# ---------------------------------------------------------------------------
# Bash sentinel and error handling
# ---------------------------------------------------------------------------


class TestBashSentinelHandling:
    """Verify heredoc commands don't corrupt the sentinel and errors are caught."""

    @pytest.fixture
    def session(self):
        s = ClaudeBashSession.__new__(ClaudeBashSession)
        s._sentinel = "<<exit>>"
        s._timeout = 5
        s._timed_out = False
        s._process = MagicMock()
        s._process.stdin = MagicMock()
        s._process.stdin.write = MagicMock()
        s._process.stdin.drain = AsyncMock()
        s._process.stdout = MagicMock()
        s._process.stderr = MagicMock()
        s._process.returncode = None
        return s

    @pytest.mark.asyncio
    async def test_heredoc_command_completes(self, session):
        """A heredoc command must complete and return output, not hang."""
        heredoc_cmd = "cat << 'EOF'\nhello world\nEOF"
        expected_output = "hello world\n"
        sentinel_line = f"{session._sentinel}\n"

        session._process.stdout.readuntil = AsyncMock(
            return_value=(expected_output + sentinel_line).encode()
        )
        session._process.stderr.read = AsyncMock(return_value=b"")

        result = await session.run(heredoc_cmd)
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_multiline_command_completes(self, session):
        """A multi-line command with continuation should complete."""
        multiline_cmd = "echo 'line1' && \\\necho 'line2'"
        expected_output = "line1\nline2\n"
        sentinel_line = f"{session._sentinel}\n"

        session._process.stdout.readuntil = AsyncMock(
            return_value=(expected_output + sentinel_line).encode()
        )
        session._process.stderr.read = AsyncMock(return_value=b"")

        result = await session.run(multiline_cmd)
        assert "line1" in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio
    async def test_sentinel_uses_newline_separator(self, session):
        """The sentinel echo must use a newline separator, not semicolon.

        When a semicolon is used ('; echo sentinel'), heredoc delimiters like
        EOF get corrupted into 'EOF; echo sentinel' and never match, causing
        the readuntil to hang until timeout. Newline separator avoids this.
        """
        test_cmd = "echo test"
        sentinel_line = f"{session._sentinel}\n"

        session._process.stdout.readuntil = AsyncMock(
            return_value=(f"test\n{sentinel_line}").encode()
        )
        session._process.stderr.read = AsyncMock(return_value=b"")

        await session.run(test_cmd)

        written = session._process.stdin.write.call_args[0][0]
        written_str = written.decode()
        assert f"\necho '{session._sentinel}'" in written_str, (
            "Sentinel echo must use newline separator (\\n), not semicolon (;)"
        )

    @pytest.mark.asyncio
    async def test_incomplete_read_handled(self, session):
        """When the bash process exits unexpectedly, IncompleteReadError
        must be caught and converted to a ToolError, not left unhandled.
        """
        session._process.stdout.readuntil = AsyncMock(
            side_effect=asyncio.IncompleteReadError(partial=b"", expected=16)
        )
        session._process.returncode = -9

        with pytest.raises(ToolError, match="bash exited unexpectedly"):
            await session.run("exit 1")
