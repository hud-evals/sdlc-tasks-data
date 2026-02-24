"""Executors for running system commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseExecutor

if TYPE_CHECKING:
    from .pyautogui import PyAutoGUIExecutor
    from .xdo import XDOExecutor

__all__ = [
    "BaseExecutor",
    "PyAutoGUIExecutor",
    "XDOExecutor",
]


def __getattr__(name: str) -> Any:
    """Lazy import executors to avoid importing pyautogui unless needed."""
    if name == "PyAutoGUIExecutor":
        from .pyautogui import PyAutoGUIExecutor

        return PyAutoGUIExecutor
    elif name == "XDOExecutor":
        from .xdo import XDOExecutor

        return XDOExecutor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
