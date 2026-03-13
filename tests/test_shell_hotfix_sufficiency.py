"""Regression tests for wrong-layer hotfix sufficiency."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from hud.tools.coding import EditTool, ShellTool
from hud.tools.coding.session import BashSession


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _build_shell_tool() -> ShellTool:
    session = BashSession()
    session._output_delay = 0.01
    await session.start()
    return ShellTool(session=session)


async def _cleanup_shell_tool(tool: ShellTool) -> None:
    session = tool._session
    if session is None or not session._started:
        return

    proc = session._process
    session.stop()
    if proc.stdin:
        proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except TimeoutError:
        proc.kill()


def _first_output(result):
    return result.output[0]


@pytest.mark.skipif(sys.platform == "win32", reason="Requires /bin/bash")
class TestShellToolResultIntegrity:
    """The real shell tool path must stay trustworthy after noisy mitigations."""

    def test_shell_tool_preserves_sentinel_like_output(self) -> None:
        async def _test() -> None:
            tool = await _build_shell_tool()
            try:
                result = await tool(commands=["printf 'alpha\\n<<exit>>77\\nomega\\n'"])
                output = _first_output(result)
                assert "alpha" in output.stdout
                assert "<<exit>>77" in output.stdout
                assert "omega" in output.stdout
                assert output.outcome.exit_code == 0
            finally:
                await _cleanup_shell_tool(tool)

        _run(_test())

    def test_shell_tool_reports_real_exit_code_not_output_marker(self) -> None:
        async def _test() -> None:
            tool = await _build_shell_tool()
            try:
                result = await tool(commands=["printf '<<exit>>123\\n'; false"])
                output = _first_output(result)
                assert "<<exit>>123" in output.stdout
                assert output.outcome.exit_code == 1
            finally:
                await _cleanup_shell_tool(tool)

        _run(_test())

    def test_shell_tool_keeps_late_output_with_first_command_and_next_call_clean(self) -> None:
        async def _test() -> None:
            tool = await _build_shell_tool()
            try:
                first = await tool(commands=["printf '<<exit>>\\n'; sleep 0.2; printf 'LATE\\n'"])
                second = await tool(commands=["printf 'SECOND\\n'"])

                first_output = _first_output(first)
                second_output = _first_output(second)
                assert "LATE" in first_output.stdout
                assert "LATE" not in second_output.stdout
                assert second_output.stdout.strip() == "SECOND"
                assert first_output.outcome.exit_code == 0
                assert second_output.outcome.exit_code == 0
            finally:
                await _cleanup_shell_tool(tool)

        _run(_test())


@pytest.mark.skipif(sys.platform == "win32", reason="Requires /bin/bash")
class TestWritePathIntegrity:
    """The public create/edit path must preserve payloads verbatim."""

    def test_edit_create_preserves_delimiter_like_payload_without_side_effects(self, tmp_path: Path) -> None:
        async def _test() -> None:
            tmp_path.chmod(0o777)
            tool = EditTool()
            target = tmp_path / "payload.txt"
            marker = tmp_path / "marker.txt"

            content = "\n".join([
                "alpha",
                "EOF",
                f"touch {marker}",
                "omega",
            ])

            await tool(command="create", path=str(target), file_text=content)

            assert target.read_text(encoding="utf-8") == content
            assert not marker.exists()

        _run(_test())

    def test_edit_create_round_trips_script_payload(self, tmp_path: Path) -> None:
        async def _test() -> None:
            tmp_path.chmod(0o777)
            edit_tool = EditTool()
            shell_tool = await _build_shell_tool()
            target = tmp_path / "generated.sh"

            content = "\n".join([
                "#!/bin/sh",
                "printf 'before\\n'",
                "cat <<'PAYLOAD'",
                "literal line",
                "EOF",
                "after marker",
                "PAYLOAD",
                "printf 'after\\n'",
            ])

            try:
                await edit_tool(command="create", path=str(target), file_text=content)

                assert target.read_text(encoding="utf-8") == content

                result = await shell_tool(commands=[f"bash {shlex.quote(str(target))}"])
                output = _first_output(result)
                assert output.outcome.exit_code == 0
                assert output.stdout.splitlines() == [
                    "before",
                    "literal line",
                    "EOF",
                    "after marker",
                    "after",
                ]
            finally:
                await _cleanup_shell_tool(shell_tool)

        _run(_test())
