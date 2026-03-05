"""Boundary-behavior tests for apply_patch workspace safety."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hud.tools.coding.apply_patch import ApplyPatchTool


@pytest.mark.asyncio
async def test_create_file_blocks_boundary_escape_via_nested_path() -> None:
    """Creating through an in-workspace alias path must not write outside."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        workspace = root / "workspace"
        outside_dir = root / "outside"
        workspace.mkdir()
        outside_dir.mkdir()

        # Alias appears in-workspace but targets a directory outside it.
        (workspace / "runtime").symlink_to(outside_dir, target_is_directory=True)

        tool = ApplyPatchTool(base_path=str(workspace))
        result = await tool(type="create_file", path="runtime/escaped.txt", diff="+owned")

        assert result.status == "failed"
        assert not (outside_dir / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_update_file_blocks_boundary_escape_via_nested_path() -> None:
    """Updating through an in-workspace alias path must not mutate outside files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        workspace = root / "workspace"
        outside_dir = root / "outside"
        workspace.mkdir()
        outside_dir.mkdir()

        outside_file = outside_dir / "target.txt"
        outside_file.write_text("line1\nline2", encoding="utf-8")

        # Alias appears in-workspace but targets a directory outside it.
        (workspace / "cache").symlink_to(outside_dir, target_is_directory=True)

        tool = ApplyPatchTool(base_path=str(workspace))
        result = await tool(
            type="update_file",
            path="cache/target.txt",
            diff=" line1\n-line2\n+lineX",
        )

        assert result.status == "failed"
        assert outside_file.read_text(encoding="utf-8") == "line1\nline2"
