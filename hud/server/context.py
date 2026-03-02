"""
HUD context helpers for persistent state across hot-reloads.

Provides utilities for creating shared context servers that survive
code reloads during development.

Usage in your environment:
    # In your context_server.py:
    from hud.server.context import serve_context

    class MyContext:
        def __init__(self):
            self.state = {}
        def startup(self):
            # Initialize resources
            pass

    if __name__ == "__main__":
        serve_context(MyContext())

    # In your MCP server:
    from hud.server.context import attach_context
    ctx = attach_context()  # Gets the persistent context
"""

from __future__ import annotations

import asyncio
import logging
import os
from multiprocessing.managers import BaseManager
from typing import Any

logger = logging.getLogger(__name__)
# Default Unix socket path (can be overridden with HUD_CTX_SOCK)
DEFAULT_SOCK_PATH = "/tmp/hud_ctx.sock"  # noqa: S108


def serve_context(
    context_instance: Any, sock_path: str | None = None, authkey: bytes = b"hud-context"
) -> BaseManager:
    """
    Serve a context object via multiprocessing Manager.

    Args:
        context_instance: The context object to serve
        sock_path: Unix socket path (defaults to HUD_CTX_SOCK env var or /tmp/hud_ctx.sock)
        authkey: Authentication key for the manager

    Returns:
        The manager instance (can be used to shutdown)
    """
    sock_path = sock_path or os.getenv("HUD_CTX_SOCK", DEFAULT_SOCK_PATH)

    class ContextManager(BaseManager):
        pass

    ContextManager.register("get_context", callable=lambda: context_instance)

    manager = ContextManager(address=sock_path, authkey=authkey)
    manager.start()

    return manager


def attach_context(sock_path: str | None = None, authkey: bytes = b"hud-context") -> Any:
    """
    Attach to a running context server.

    Args:
        sock_path: Unix socket path (defaults to HUD_CTX_SOCK env var or /tmp/hud_ctx.sock)
        authkey: Authentication key for the manager

    Returns:
        The shared context object
    """
    sock_path = sock_path or os.getenv("HUD_CTX_SOCK", DEFAULT_SOCK_PATH)

    class ContextManager(BaseManager):
        pass

    ContextManager.register("get_context")

    manager = ContextManager(address=sock_path, authkey=authkey)
    manager.connect()

    return manager.get_context()  # type: ignore


async def run_context_server(
    context_instance: Any, sock_path: str | None = None, authkey: bytes = b"hud-context"
) -> None:
    """
    Run a context server until interrupted.

    Args:
        context_instance: The context object to serve
        sock_path: Unix socket path
        authkey: Authentication key
    """
    sock_path = sock_path or os.getenv("HUD_CTX_SOCK", DEFAULT_SOCK_PATH)

    logger.info("[Context Server] Starting on %s...", sock_path)

    # Start the manager
    manager = serve_context(context_instance, sock_path, authkey)
    logger.info("[Context Server] Ready on %s", sock_path)

    # Wait forever (until killed)
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("[Context Server] Shutting down...")
        manager.shutdown()
