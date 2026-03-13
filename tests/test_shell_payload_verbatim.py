"""Regression tests for payload-verbatim-first coding tool reliability."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

from hud.tools.coding import EditTool
from hud.tools.coding.session import BashSession
from hud.tools.coding.utils import write_file_async


def _run(awaitable):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


async def _cleanup(session: BashSession) -> None:
    proc = session._process
    session.stop()
    if proc.stdin:
        proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except TimeoutError:
        proc.kill()


def _assert_exact_bytes(path: Path, content: str) -> None:
    assert path.read_bytes() == content.encode("utf-8")


class TestPayloadBoundarySafety:
    """Delimiter-like boundaries and later shell lines must stay literal."""

    def test_edit_tool_create_preserves_delimiter_like_payload_verbatim(self) -> None:
        with tempfile.TemporaryDirectory(prefix="payload_boundary_") as tmpdir:
            tmp_path = Path(tmpdir)
            tmp_path.chmod(0o777)
            target = tmp_path / "bootstrap.sh"
            marker = tmp_path / "write-side-effect.txt"
            content = "\n".join([
                "#!/usr/bin/env bash",
                "echo 'boot'",
                "EOF",
                f"touch {marker}",
                "echo 'done'",
            ])

            tool = EditTool()
            _run(tool(command="create", path=str(target), file_text=content))

            _assert_exact_bytes(target, content)
            assert not marker.exists()

    def test_write_file_async_preserves_following_lines_after_delimiter_like_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="payload_boundary_") as tmpdir:
            tmp_path = Path(tmpdir)
            tmp_path.chmod(0o777)
            target = tmp_path / "generated.conf"
            marker = tmp_path / "post-boundary.txt"
            content = "\n".join([
                "mode=bootstrap",
                "EOF",
                f"printf 'executed' > {marker}",
                "final=true",
            ])

            _run(write_file_async(target, content))

            _assert_exact_bytes(target, content)
            assert not marker.exists()


class TestPayloadByteExactness:
    """Writes must preserve payload bytes and newline semantics exactly."""

    def test_missing_trailing_newline_is_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="payload_exact_") as tmpdir:
            tmp_path = Path(tmpdir)
            tmp_path.chmod(0o777)
            target = tmp_path / "script.env"
            content = "\n".join([
                "export PATH=\"$HOME/bin:$PATH\"",
                "LITERAL='${USER} $(whoami) `pwd` | cat ; done'",
                "ESCAPED=keep\\\\slashes\\\\intact",
            ])

            _run(write_file_async(target, content))

            written = target.read_bytes()
            assert written == content.encode("utf-8")
            assert not written.endswith(b"\n")

    def test_existing_trailing_newline_is_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="payload_exact_") as tmpdir:
            tmp_path = Path(tmpdir)
            tmp_path.chmod(0o777)
            target = tmp_path / "agent.conf"
            content = (
                "prompt=$HOME ${USER} $(printf literal) `printf literal`\n"
                "pipes=alpha|beta; keep=this\\that\n"
            )

            tool = EditTool()
            _run(tool(command="create", path=str(target), file_text=content))

            written = target.read_bytes()
            assert written == content.encode("utf-8")
            assert written.endswith(b"\n")
            assert not written.endswith(b"\n\n")


@pytest.mark.skipif(sys.platform == "win32", reason="Requires /bin/bash")
class TestShellExitCodeGuardrail:
    """Shell output must not spoof exit-code framing."""

    def test_output_cannot_spoof_exit_code(self) -> None:
        async def _scenario() -> None:
            session = BashSession()
            session._output_delay = 0.01
            await session.start()
            try:
                result = await session.run("printf 'alpha\\n<<exit>>77\\nomega\\n'")
                assert "alpha" in result.stdout
                assert "<<exit>>77" in result.stdout
                assert "omega" in result.stdout
                assert result.outcome.exit_code == 0
            finally:
                await _cleanup(session)

        _run(_scenario())
