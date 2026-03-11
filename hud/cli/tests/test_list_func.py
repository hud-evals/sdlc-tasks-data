"""Tests for list_func.py - List HUD environments from local registry."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest
import yaml

from hud.cli.list_func import format_timestamp, list_command, list_environments


class TestFormatTimestamp:
    """Test timestamp formatting functionality."""

    def test_format_timestamp_just_now(self):
        """Test formatting for very recent timestamps."""
        now = time.time()
        assert format_timestamp(now) == "just now"

    def test_format_timestamp_minutes(self):
        """Test formatting for timestamps in minutes."""
        now = datetime.now()

        # 5 minutes ago
        timestamp = (now - timedelta(minutes=5)).timestamp()
        assert format_timestamp(timestamp) == "5m ago"

        # 45 minutes ago
        timestamp = (now - timedelta(minutes=45)).timestamp()
        assert format_timestamp(timestamp) == "45m ago"

    def test_format_timestamp_hours(self):
        """Test formatting for timestamps in hours."""
        now = datetime.now()

        # 2 hours ago
        timestamp = (now - timedelta(hours=2)).timestamp()
        assert format_timestamp(timestamp) == "2h ago"

        # 23 hours ago
        timestamp = (now - timedelta(hours=23)).timestamp()
        assert format_timestamp(timestamp) == "23h ago"

    def test_format_timestamp_days(self):
        """Test formatting for timestamps in days."""
        now = datetime.now()

        # 3 days ago
        timestamp = (now - timedelta(days=3)).timestamp()
        assert format_timestamp(timestamp) == "3d ago"

        # 29 days ago
        timestamp = (now - timedelta(days=29)).timestamp()
        assert format_timestamp(timestamp) == "29d ago"

    def test_format_timestamp_months(self):
        """Test formatting for timestamps in months."""
        now = datetime.now()

        # 2 months ago
        timestamp = (now - timedelta(days=60)).timestamp()
        assert format_timestamp(timestamp) == "2mo ago"

        # 11 months ago
        timestamp = (now - timedelta(days=335)).timestamp()
        assert format_timestamp(timestamp) == "11mo ago"

    def test_format_timestamp_years(self):
        """Test formatting for timestamps in years."""
        now = datetime.now()

        # 1 year ago
        timestamp = (now - timedelta(days=400)).timestamp()
        assert format_timestamp(timestamp) == "1y ago"

        # 3 years ago
        timestamp = (now - timedelta(days=1100)).timestamp()
        assert format_timestamp(timestamp) == "3y ago"

    def test_format_timestamp_none(self):
        """Test formatting when timestamp is None."""
        assert format_timestamp(None) == "unknown"


class TestListEnvironments:
    """Test listing environments functionality."""

    @pytest.fixture
    def mock_registry_dir(self, tmp_path):
        """Create a mock registry directory with sample environments."""
        registry_dir = tmp_path / ".hud" / "envs"
        registry_dir.mkdir(parents=True)

        # Create sample environments (use underscore instead of colon for Windows compatibility)
        env1_dir = registry_dir / "sha256_abc123"
        env1_dir.mkdir()
        lock1 = env1_dir / "hud.lock.yaml"
        lock1.write_text(
            yaml.dump(
                {
                    "image": "test/env1:latest",
                    "metadata": {
                        "description": "Test environment 1",
                        "tools": ["tool1", "tool2", "tool3"],
                    },
                }
            )
        )
        # Set modification time to 1 hour ago
        lock1.touch()

        env2_dir = registry_dir / "sha256_def456"
        env2_dir.mkdir()
        lock2 = env2_dir / "hud.lock.yaml"
        lock2.write_text(
            yaml.dump(
                {
                    "image": "test/env2:v1.0",
                    "metadata": {
                        "description": "Test environment 2 with a much longer description that should be truncated",  # noqa: E501
                        "tools": ["tool1"],
                    },
                }
            )
        )

        return registry_dir

    @mock.patch("hud.cli.list_func.get_registry_dir")
    def test_list_no_registry(self, mock_get_registry):
        """Test listing when registry doesn't exist."""
        mock_get_registry.return_value = Path("/nonexistent")

        # Just test it runs without error
        list_environments()

        # The design messages will be printed but we don't need to test them

    @mock.patch("hud.cli.list_func.get_registry_dir")
    def test_list_json_no_registry(self, mock_get_registry, capsys):
        """Test JSON output when registry doesn't exist."""
        mock_get_registry.return_value = Path("/nonexistent")

        list_environments(json_output=True)

        captured = capsys.readouterr()
        assert captured.out.strip() == "[]"

    @mock.patch("hud.cli.list_func.get_registry_dir")
    @mock.patch("hud.cli.list_func.list_registry_entries")
    def test_list_environments_basic(self, mock_list_entries, mock_get_registry, mock_registry_dir):
        """Test basic environment listing."""
        mock_get_registry.return_value = mock_registry_dir

        # Mock registry entries
        entries = [
            ("sha256_abc123", mock_registry_dir / "sha256_abc123" / "hud.lock.yaml"),
            ("sha256_def456", mock_registry_dir / "sha256_def456" / "hud.lock.yaml"),
        ]
        mock_list_entries.return_value = entries

        # Just test it runs without error
        list_environments()

    @mock.patch("hud.cli.list_func.get_registry_dir")
    @mock.patch("hud.cli.list_func.list_registry_entries")
    def test_list_environments_json(
        self, mock_list_entries, mock_get_registry, mock_registry_dir, capsys
    ):
        """Test JSON output format."""
        mock_get_registry.return_value = mock_registry_dir

        # Mock registry entries
        entries = [
            ("sha256_abc123", mock_registry_dir / "sha256_abc123" / "hud.lock.yaml"),
            ("sha256_def456", mock_registry_dir / "sha256_def456" / "hud.lock.yaml"),
        ]
        mock_list_entries.return_value = entries

        list_environments(json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert len(data) == 2
        # Check we have the expected structure
        assert all(key in data[0] for key in ["name", "tag", "tools_count", "digest"])

    @mock.patch("hud.cli.list_func.get_registry_dir")
    @mock.patch("hud.cli.list_func.list_registry_entries")
    def test_list_environments_filter(
        self, mock_list_entries, mock_get_registry, mock_registry_dir, capsys
    ):
        """Test filtering environments by name."""
        mock_get_registry.return_value = mock_registry_dir

        # Mock registry entries
        entries = [
            ("sha256_abc123", mock_registry_dir / "sha256_abc123" / "hud.lock.yaml"),
            ("sha256_def456", mock_registry_dir / "sha256_def456" / "hud.lock.yaml"),
        ]
        mock_list_entries.return_value = entries

        # Filter for env1 and check JSON output
        list_environments(filter_name="env1", json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Should only have env1
        assert len(data) == 1
        assert "env1" in data[0]["name"]

    @mock.patch("hud.cli.list_func.get_registry_dir")
    @mock.patch("hud.cli.list_func.list_registry_entries")
    def test_list_environments_verbose(
        self, mock_list_entries, mock_get_registry, mock_registry_dir
    ):
        """Test verbose output."""
        mock_get_registry.return_value = mock_registry_dir

        # Mock registry entries
        entries = [
            ("sha256_abc123", mock_registry_dir / "sha256_abc123" / "hud.lock.yaml"),
        ]
        mock_list_entries.return_value = entries

        # Just test it runs in verbose mode
        list_environments(verbose=True)

    @mock.patch("hud.cli.list_func.get_registry_dir")
    @mock.patch("hud.cli.list_func.list_registry_entries")
    def test_list_environments_with_errors(
        self, mock_list_entries, mock_get_registry, mock_registry_dir, capsys
    ):
        """Test handling of corrupted lock files."""
        mock_get_registry.return_value = mock_registry_dir

        # Create a bad lock file
        bad_dir = mock_registry_dir / "sha256_bad"
        bad_dir.mkdir()
        bad_lock = bad_dir / "hud.lock.yaml"
        bad_lock.write_text("invalid: yaml: content:")

        # Mock registry entries including the bad one
        entries = [
            ("sha256_bad", bad_lock),
            ("sha256_abc123", mock_registry_dir / "sha256_abc123" / "hud.lock.yaml"),
            ("sha256_def456", mock_registry_dir / "sha256_def456" / "hud.lock.yaml"),
        ]
        mock_list_entries.return_value = entries

        # Should handle error gracefully in verbose mode
        list_environments(verbose=True, json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Should still list the valid environments
        assert len(data) == 2  # Only the 2 valid ones

    @mock.patch("hud.cli.list_func.get_registry_dir")
    @mock.patch("hud.cli.list_func.list_registry_entries")
    def test_list_environments_no_matches(
        self, mock_list_entries, mock_get_registry, mock_registry_dir
    ):
        """Test when no environments match the filter."""
        mock_get_registry.return_value = mock_registry_dir

        # Mock registry entries
        entries = [
            ("sha256_abc123", mock_registry_dir / "sha256_abc123" / "hud.lock.yaml"),
            ("sha256_def456", mock_registry_dir / "sha256_def456" / "hud.lock.yaml"),
        ]
        mock_list_entries.return_value = entries

        # Filter for non-existent env
        list_environments(filter_name="nonexistent")

        # Just test it runs without error


class TestListCommand:
    """Test the CLI command wrapper."""

    def test_list_command_basic(self):
        """Test basic list command runs without error."""
        # Just test it doesn't crash
        # Note: we can't easily test the exact arguments because Typer
        # passes OptionInfo objects as defaults
        list_command()

    def test_list_command_with_options(self):
        """Test list command with options runs without error."""
        # Just test it doesn't crash with explicit values
        list_command(filter_name="test", json_output=True, show_all=True, verbose=True)
