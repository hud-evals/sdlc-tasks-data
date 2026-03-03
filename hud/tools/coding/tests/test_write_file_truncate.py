"""Regression tests for async file writing behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hud.tools.coding.utils import write_file_async


@pytest.mark.asyncio
async def test_write_file_async_treats_eof_and_shell_like_lines_as_content():
    """Writing content should not interpret payload lines as shell syntax."""
    with tempfile.TemporaryDirectory(prefix="write_file_async_") as tmpdir:
        tmp_path = Path(tmpdir)
        target_file = tmp_path / "payload.txt"
        marker_file = tmp_path / "marker.txt"

        content = "\n".join([
            "alpha",
            "EOF",
            f"touch {marker_file}",
            "omega",
        ])

        await write_file_async(target_file, content)

        assert target_file.read_text(encoding="utf-8") == content
        assert not marker_file.exists()
