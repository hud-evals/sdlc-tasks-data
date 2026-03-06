"""Hidden tests for routing cache coherence SRE task.

Every test here FAILS on the baseline and PASSES on the golden fix.
Tests verify the actual behavioral bugs, not just attribute existence.
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock


class TestListPromptsRebuildsRouting:
    """list_prompts() must call _build_prompt_routing after refreshing caches."""

    def test_list_prompts_source_contains_build_call(self):
        """Static check: list_prompts source must contain _build_prompt_routing."""
        from hud.environment import Environment

        source = inspect.getsource(Environment.list_prompts)
        assert "_build_prompt_routing" in source, (
            "list_prompts() does not call _build_prompt_routing(). "
            "It refreshes connector caches but never rebuilds the router's "
            "prompt lookup table, leaving stale routing after reconnection."
        )

    def test_list_prompts_calls_build_routing(self):
        """Behavioral check: list_prompts must invoke _build_prompt_routing."""
        from hud.environment import Environment

        env = Environment("test")
        env._connections = {}
        env._build_prompt_routing = AsyncMock()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(env.list_prompts())
        finally:
            loop.close()

        env._build_prompt_routing.assert_awaited_once(), (
            "list_prompts() did not call _build_prompt_routing(). "
            "Connector caches were refreshed but the router's prompt "
            "routing table was not rebuilt."
        )


class TestListResourcesRebuildsRouting:
    """list_resources() must call _build_resource_routing after refreshing caches."""

    def test_list_resources_source_contains_build_call(self):
        """Static check: list_resources source must contain _build_resource_routing."""
        from hud.environment import Environment

        source = inspect.getsource(Environment.list_resources)
        assert "_build_resource_routing" in source, (
            "list_resources() does not call _build_resource_routing(). "
            "It refreshes connector caches but never rebuilds the router's "
            "resource lookup table, leaving stale routing after reconnection."
        )

    def test_list_resources_calls_build_routing(self):
        """Behavioral check: list_resources must invoke _build_resource_routing."""
        from hud.environment import Environment

        env = Environment("test")
        env._connections = {}
        env._build_resource_routing = AsyncMock()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(env.list_resources())
        finally:
            loop.close()

        env._build_resource_routing.assert_awaited_once(), (
            "list_resources() did not call _build_resource_routing(). "
            "Connector caches were refreshed but the router's resource "
            "routing table was not rebuilt."
        )


class TestActiveSessionCleared:
    """__aexit__ must clear _active_session to prevent stale scenario state."""

    def test_aexit_clears_active_session(self):
        from hud.environment import Environment

        env = Environment("test")
        env._active_session = MagicMock()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(env.__aexit__(None, None, None))
        finally:
            loop.close()

        assert env._active_session is None, (
            "_active_session not cleared in __aexit__. "
            "Stale scenario state will persist across reconnections."
        )


class TestConnectorReconnect:
    """Connector must have a reconnect() method for clean lifecycle management."""

    def test_connector_has_reconnect(self):
        from hud.environment.connection import Connector

        assert hasattr(Connector, "reconnect"), (
            "Connector has no reconnect() method. Callers must manually "
            "orchestrate disconnect/connect/refetch, which is error-prone "
            "and leads to stale caches."
        )

    def test_reconnect_is_async_callable(self):
        from hud.environment.connection import Connector

        assert hasattr(Connector, "reconnect"), (
            "Connector.reconnect does not exist"
        )
        assert asyncio.iscoroutinefunction(Connector.reconnect), (
            "Connector.reconnect must be an async method"
        )
