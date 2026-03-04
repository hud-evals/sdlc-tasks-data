"""Regression tests for shell command stream framing/isolation."""

from __future__ import annotations

import asyncio
import sys

import pytest

from hud.tools.coding.session import BashSession


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
class TestShellCommandStreamIsolation:
    """Integration tests around command output framing boundaries."""

    @pytest.mark.asyncio
    async def test_marker_like_user_output_is_preserved(self) -> None:
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
    async def test_exit_code_is_not_spoofed_by_output(self) -> None:
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
    async def test_no_cross_command_output_bleed(self) -> None:
        session = BashSession()
        session._output_delay = 0.01
        await session.start()
        try:
            first = await session.run("printf '<<exit>>\\n'; sleep 0.2; printf 'LATE\\n'")
            second = await session.run("printf 'SECOND\\n'")

            assert "LATE" in first.stdout
            assert "LATE" not in second.stdout
            assert second.stdout.strip() == "SECOND"
            assert first.outcome.exit_code == 0
            assert second.outcome.exit_code == 0
        finally:
            await _cleanup(session)
