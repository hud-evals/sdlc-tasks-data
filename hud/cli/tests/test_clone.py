"""Tests for the clone command."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, mock_open, patch

from hud.cli.clone import clone_repository, get_clone_message


def test_clone_repository_success():
    """Test successful repository cloning."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        success, result = clone_repository("https://github.com/user/repo.git")

        assert success is True
        assert "repo" in result
        mock_run.assert_called_once()

        # Check command includes quiet flag
        cmd = mock_run.call_args[0][0]
        assert "git" in cmd
        assert "clone" in cmd
        assert "--quiet" in cmd
        assert "https://github.com/user/repo.git" in cmd


def test_clone_repository_failure():
    """Test failed repository cloning."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            128, ["git", "clone"], stderr="fatal: repository not found"
        )

        success, result = clone_repository("https://github.com/user/nonexistent.git")

        assert success is False
        assert "repository not found" in result


def test_get_clone_message_from_pyproject():
    """Test reading clone message from pyproject.toml."""
    toml_content = """
[tool.hud.clone]
title = "Test Project"
message = "Welcome to the test project!"
"""

    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("builtins.open", mock_open(read_data=toml_content.encode())),
        patch("tomllib.load") as mock_load,
    ):
        mock_exists.return_value = True
        mock_load.return_value = {
            "tool": {
                "hud": {
                    "clone": {"title": "Test Project", "message": "Welcome to the test project!"}
                }
            }
        }

        config = get_clone_message("/path/to/repo")

        assert config is not None
        assert config["title"] == "Test Project"
        assert config["message"] == "Welcome to the test project!"


def test_get_clone_message_from_hud_toml():
    """Test reading clone message from .hud.toml."""
    toml_content = """
[clone]
title = "HUD Project"
markdown = "## Welcome!"
style = "cyan"
"""

    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("builtins.open", mock_open(read_data=toml_content.encode())),
        patch("tomllib.load") as mock_load,
    ):
        # First call for pyproject.toml returns False
        # Second call for .hud.toml returns True
        mock_exists.side_effect = [False, True]
        mock_load.return_value = {
            "clone": {"title": "HUD Project", "markdown": "## Welcome!", "style": "cyan"}
        }

        config = get_clone_message("/path/to/repo")

        assert config is not None
        assert config["title"] == "HUD Project"
        assert config["markdown"] == "## Welcome!"
        assert config["style"] == "cyan"


def test_get_clone_message_none():
    """Test when no clone message configuration exists."""
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False

        config = get_clone_message("/path/to/repo")

        assert config is None


# The following tests are commented out as print_success and print_error
# functions are no longer part of the clone module

# def test_print_success(capsys):
#     """Test success message printing."""
#     print_success("https://github.com/user/repo.git", "/home/user/repo")

#     captured = capsys.readouterr()
#     assert "Successfully cloned" in captured.out
#     assert "repo" in captured.out
#     assert "/home/user/repo" in captured.out


# def test_print_success_with_config(capsys):
#     """Test success message with configuration."""
#     config = {"title": "My Project", "message": "Thanks for cloning!"}

#     print_success("https://github.com/user/repo.git", "/home/user/repo", config)

#     captured = capsys.readouterr()
#     assert "Successfully cloned" in captured.out
#     assert "My Project" in captured.out
#     assert "Thanks for cloning!" in captured.out


# def test_print_error(capsys):
#     """Test error message printing."""
#     print_error("Repository not found")

#     captured = capsys.readouterr()
#     assert "Repository not found" in captured.out
#     assert "Clone Failed" in captured.out
