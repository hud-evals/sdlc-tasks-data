"""Tests for MCP response reliability: exception propagation, timeout
configuration, and transport-level retry behavior."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_http_status_error(status_code: int = 502):
    """Create a realistic httpx.HTTPStatusError for testing."""
    try:
        import httpx
        request = httpx.Request("POST", "https://mcp.hud.ai/v3/mcp")
        response = httpx.Response(status_code, request=request)
        return httpx.HTTPStatusError(
            f"Server error '{status_code}'",
            request=request,
            response=response,
        )
    except ImportError:
        err = Exception(f"Server error '{status_code}'")
        err.response = MagicMock(status_code=status_code)
        return err


# ---------------------------------------------------------------------------
# 1. JSON response exception propagation
# ---------------------------------------------------------------------------

class TestJsonResponseExceptionPropagation:
    """After applying patches, _handle_json_response must re-raise exceptions
    so callers get proper errors instead of hanging indefinitely waiting for
    a response future that is never fulfilled."""

    def test_parse_error_propagates_to_caller(self):
        """When _handle_json_response receives a response with invalid JSON,
        the ValidationError should propagate to the caller — not be silently
        sent to the read stream where it gets dropped."""
        from hud.patches.mcp_patches import apply_all_patches
        apply_all_patches()

        from mcp.client.streamable_http import StreamableHTTPTransport

        class BadJsonResponse:
            async def aread(self) -> bytes:
                return b"{{invalid json content"

        class NoopWriter:
            async def send(self, msg: object) -> None:
                pass

        async def _test():
            with pytest.raises(Exception):
                await StreamableHTTPTransport._handle_json_response(
                    object(), BadJsonResponse(), NoopWriter(),
                    is_initialization=False,
                )

        _run(_test())

    def test_read_error_propagates_to_caller(self):
        """When response.aread() raises a read error (e.g., connection reset),
        the error should propagate to the caller — not be silently swallowed."""
        from hud.patches.mcp_patches import apply_all_patches
        apply_all_patches()

        from mcp.client.streamable_http import StreamableHTTPTransport

        class ReadErrorResponse:
            async def aread(self) -> bytes:
                raise IOError("Connection reset by peer during response read")

        class NoopWriter:
            async def send(self, msg: object) -> None:
                pass

        async def _test():
            with pytest.raises((IOError, Exception)):
                await StreamableHTTPTransport._handle_json_response(
                    object(), ReadErrorResponse(), NoopWriter(),
                    is_initialization=False,
                )

        _run(_test())


# ---------------------------------------------------------------------------
# 2. Timeout configuration separation
# ---------------------------------------------------------------------------

class TestTimeoutConfiguration:
    """client_timeout (total session lifetime) and sse_read_timeout
    (per-message read timeout) must be independently configurable in
    settings. A value of 0 for sse_read_timeout must mean 'no timeout',
    not a literal 0-second timeout that fails instantly."""

    def test_settings_has_separate_sse_read_timeout_field(self):
        """Settings must expose sse_read_timeout as a field that is
        independent from client_timeout."""
        from hud.settings import Settings
        fields = Settings.model_fields
        assert "sse_read_timeout" in fields, (
            "Settings must define sse_read_timeout as a separate field "
            "from client_timeout"
        )

    def test_connect_mcp_uses_sse_read_timeout_setting_not_client_timeout(self):
        """connect_mcp() must pass settings.sse_read_timeout through to the
        transport instead of deriving the transport timeout from
        settings.client_timeout."""
        with patch(
            "hud.environment.connectors.mcp_config._build_transport"
        ) as mock_build, patch(
            "hud.settings.settings"
        ) as mock_settings:
            mock_settings.client_timeout = 900
            mock_settings.sse_read_timeout = 37
            mock_build.return_value = MagicMock()

            from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

            mixin = MCPConfigConnectorMixin.__new__(MCPConfigConnectorMixin)
            mixin._add_connection = MagicMock()

            mixin.connect_mcp(
                {"test": {"url": "https://mcp.hud.ai/v3/mcp"}}
            )

            built_config = mock_build.call_args[0][0]
            assert built_config["sse_read_timeout"] == 37, (
                "connect_mcp() must pass settings.sse_read_timeout through to the "
                "transport instead of deriving it from settings.client_timeout"
            )

    def test_zero_sse_timeout_becomes_none(self):
        """When sse_read_timeout=0 (meaning 'no timeout'), connect_mcp
        must NOT pass a literal 0 to the transport. It should convert 0
        to None so the transport has no read timeout."""
        with patch(
            "hud.environment.connectors.mcp_config._build_transport"
        ) as mock_build, patch(
            "hud.settings.settings"
        ) as mock_settings:
            mock_settings.sse_read_timeout = 0
            mock_settings.client_timeout = 900
            mock_build.return_value = MagicMock()

            from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

            mixin = MCPConfigConnectorMixin.__new__(MCPConfigConnectorMixin)
            mixin._add_connection = MagicMock()

            mixin.connect_mcp(
                {"test": {"url": "https://mcp.hud.ai/v3/mcp"}}
            )

            built_config = mock_build.call_args[0][0]
            timeout = built_config.get("sse_read_timeout")
            assert timeout is None, (
                "sse_read_timeout=0 must mean 'no read timeout' and be converted "
                f"to None before constructing the transport, but got {timeout}"
            )


# ---------------------------------------------------------------------------
# 3. Transport-level 5xx retry
# ---------------------------------------------------------------------------

class TestTransportRetryOn5xx:
    """Transport-level 502/503/504 HTTP status errors must be retried
    with backoff by the post_writer retry loop. Non-retryable errors
    (e.g., 400) should propagate immediately without retry."""

    def test_502_triggers_retry(self):
        """When _handle_post_request raises HTTPStatusError(502), the
        patched post_writer retry loop should retry instead of failing
        immediately."""
        import httpx
        from hud.patches.mcp_patches import apply_all_patches
        apply_all_patches()

        from mcp.client.streamable_http import StreamableHTTPTransport
        from mcp.types import JSONRPCMessage, JSONRPCRequest
        from mcp.shared.message import SessionMessage

        transport = StreamableHTTPTransport.__new__(StreamableHTTPTransport)
        transport.session_id = "test"
        transport.request_headers = {}
        transport.sse_read_timeout = 30
        transport._is_initialized_notification = lambda msg: False

        call_count = 0

        async def mock_handle_post(ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_http_status_error(502)

        transport._handle_post_request = mock_handle_post

        request_msg = JSONRPCMessage(JSONRPCRequest(
            jsonrpc="2.0", id=1, method="tools/call",
            params={"name": "test_tool", "arguments": {}},
        ))
        session_msg = SessionMessage(request_msg)

        async def _test():
            import anyio

            write_send, write_recv = anyio.create_memory_object_stream(10)
            read_send, read_recv = anyio.create_memory_object_stream(10)

            await write_send.send(session_msg)
            await write_send.aclose()

            async with anyio.create_task_group() as tg:
                try:
                    await transport.post_writer(
                        MagicMock(), write_recv, read_send,
                        write_send, lambda: None, tg,
                    )
                except Exception:
                    pass

        from hud.settings import settings
        original = settings.client_timeout
        settings.client_timeout = 10
        try:
            _run(asyncio.wait_for(_test(), timeout=15))
        except asyncio.TimeoutError:
            pass
        finally:
            settings.client_timeout = original

        assert call_count >= 2, (
            f"502 should trigger retry, but _handle_post_request was called "
            f"only {call_count} time(s)"
        )

    def test_400_not_retried(self):
        """A 400 Bad Request should NOT be retried — it should propagate
        immediately via send_error_response."""
        import httpx
        from hud.patches.mcp_patches import apply_all_patches
        apply_all_patches()

        from mcp.client.streamable_http import StreamableHTTPTransport
        from mcp.types import JSONRPCMessage, JSONRPCRequest
        from mcp.shared.message import SessionMessage

        transport = StreamableHTTPTransport.__new__(StreamableHTTPTransport)
        transport.session_id = "test"
        transport.request_headers = {}
        transport.sse_read_timeout = 30
        transport._is_initialized_notification = lambda msg: False

        call_count = 0

        async def mock_handle_post(ctx):
            nonlocal call_count
            call_count += 1
            raise _make_http_status_error(400)

        transport._handle_post_request = mock_handle_post

        request_msg = JSONRPCMessage(JSONRPCRequest(
            jsonrpc="2.0", id=1, method="tools/call",
            params={"name": "test_tool", "arguments": {}},
        ))
        session_msg = SessionMessage(request_msg)

        async def _test():
            import anyio

            write_send, write_recv = anyio.create_memory_object_stream(10)
            read_send, read_recv = anyio.create_memory_object_stream(10)

            await write_send.send(session_msg)
            await write_send.aclose()

            async with anyio.create_task_group() as tg:
                try:
                    await transport.post_writer(
                        MagicMock(), write_recv, read_send,
                        write_send, lambda: None, tg,
                    )
                except Exception:
                    pass

        from hud.settings import settings
        original = settings.client_timeout
        settings.client_timeout = 5
        try:
            _run(asyncio.wait_for(_test(), timeout=10))
        except Exception:
            pass
        finally:
            settings.client_timeout = original

        assert call_count == 1, (
            f"400 should NOT be retried, but _handle_post_request was called "
            f"{call_count} time(s)"
        )

    def test_503_is_retried(self):
        """503 Service Unavailable should also trigger retry (same as 502)."""
        import httpx
        from hud.patches.mcp_patches import apply_all_patches
        apply_all_patches()

        from mcp.client.streamable_http import StreamableHTTPTransport
        from mcp.types import JSONRPCMessage, JSONRPCRequest
        from mcp.shared.message import SessionMessage

        transport = StreamableHTTPTransport.__new__(StreamableHTTPTransport)
        transport.session_id = "test"
        transport.request_headers = {}
        transport.sse_read_timeout = 30
        transport._is_initialized_notification = lambda msg: False

        call_count = 0

        async def mock_handle_post(ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_http_status_error(503)

        transport._handle_post_request = mock_handle_post

        request_msg = JSONRPCMessage(JSONRPCRequest(
            jsonrpc="2.0", id=1, method="tools/call",
            params={"name": "test_tool", "arguments": {}},
        ))
        session_msg = SessionMessage(request_msg)

        async def _test():
            import anyio

            write_send, write_recv = anyio.create_memory_object_stream(10)
            read_send, read_recv = anyio.create_memory_object_stream(10)

            await write_send.send(session_msg)
            await write_send.aclose()

            async with anyio.create_task_group() as tg:
                try:
                    await transport.post_writer(
                        MagicMock(), write_recv, read_send,
                        write_send, lambda: None, tg,
                    )
                except Exception:
                    pass

        from hud.settings import settings
        original = settings.client_timeout
        settings.client_timeout = 10
        try:
            _run(asyncio.wait_for(_test(), timeout=15))
        except asyncio.TimeoutError:
            pass
        finally:
            settings.client_timeout = original

        assert call_count >= 2, (
            f"503 should trigger retry, but _handle_post_request was called "
            f"only {call_count} time(s)"
        )
