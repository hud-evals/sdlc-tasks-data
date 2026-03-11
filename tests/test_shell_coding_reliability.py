"""Regression tests for coding tool reliability — shell exit codes and file writes."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

from hud.tools.coding.session import BashSession
from hud.tools.coding.utils import write_file_async


async def _cleanup(session: BashSession) -> None:
    """Cleanly shut down bash session to avoid transport warnings."""
    proc = session._process
    session.stop()
    if proc.stdin:
        proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except TimeoutError:
        proc.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="Requires /bin/bash")
class TestShellExitCodeIsolation:
    """Exit code capture must not be spoofed by command output content."""

    @pytest.mark.asyncio
    async def test_sentinel_like_output_preserved_in_stdout(self) -> None:
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

    @pytest.mark.asyncio
    async def test_exit_code_not_spoofed_by_output_content(self) -> None:
        session = BashSession()
        session._output_delay = 0.01
        await session.start()
        try:
            result = await session.run("printf '<<exit>>123\\n'; false")
            assert "<<exit>>123" in result.stdout
            assert result.outcome.exit_code == 1
        finally:
            await _cleanup(session)

    @pytest.mark.asyncio
    async def test_no_output_bleed_across_commands(self) -> None:
        session = BashSession()
        session._output_delay = 0.01
        await session.start()
        try:
            first = await session.run(
                "printf '<<exit>>\\n'; sleep 0.2; printf 'LATE\\n'"
            )
            second = await session.run("printf 'SECOND\\n'")

            assert "LATE" in first.stdout
            assert "LATE" not in second.stdout
            assert second.stdout.strip() == "SECOND"
            assert first.outcome.exit_code == 0
            assert second.outcome.exit_code == 0
        finally:
            await _cleanup(session)


class TestFileWriteContentIntegrity:
    """File writes must preserve content verbatim regardless of payload."""

    @pytest.mark.asyncio
    async def test_eof_delimiter_treated_as_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="write_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            tmp_path.chmod(0o777)
            target = tmp_path / "payload.txt"
            marker = tmp_path / "marker.txt"

            content = "\n".join([
                "alpha",
                "EOF",
                f"touch {marker}",
                "omega",
            ])

            await write_file_async(target, content)

            assert target.read_text(encoding="utf-8") == content
            assert not marker.exists()
