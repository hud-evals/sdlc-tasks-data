"""Behavioral tests for MCP response reliability."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes | None = None,
        read_error: Exception | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body or b""
        self._read_error = read_error
        self.headers = headers or {"content-type": "application/json"}

    async def aread(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body

    async def aclose(self) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return

        import httpx

        request = httpx.Request("POST", "https://mcp.hud.ai/v3/mcp")
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError(
            f"Server error '{self.status_code}'",
            request=request,
            response=response,
        )


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def stream(self, method: str, url: str, json=None, headers=None):  # noqa: A002
        if self.calls >= len(self._responses):
            raise AssertionError("No fake responses left for transport request")
        response = self._responses[self.calls]
        self.calls += 1
        return _FakeStreamContext(response)


def _success_response(request_id: int = 0) -> _FakeResponse:
    return _FakeResponse(
        body=json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {}}).encode(),
        headers={"content-type": "application/json"},
    )


@contextmanager
def _patched_runtime_settings(*, client_timeout: int, sse_read_timeout: int):
    import hud.settings as settings_module

    fake_settings = SimpleNamespace(
        client_timeout=client_timeout,
        sse_read_timeout=sse_read_timeout,
    )

    with (
        patch.dict(
            os.environ,
            {
                "HUD_CLIENT_TIMEOUT": str(client_timeout),
                "HUD_SSE_READ_TIMEOUT": str(sse_read_timeout),
            },
            clear=False,
        ),
        patch.object(settings_module, "settings", fake_settings),
        patch.object(settings_module, "get_settings", return_value=fake_settings),
    ):
        yield fake_settings


async def _send_ping_through_transport(
    responses: list[_FakeResponse],
    *,
    request_timeout_seconds: float = 0.2,
    client_timeout_seconds: float = 2.0,
):
    import anyio
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import StreamableHTTPTransport

    from hud.patches.mcp_patches import apply_all_patches
    from hud.settings import settings

    apply_all_patches()

    transport = StreamableHTTPTransport.__new__(StreamableHTTPTransport)
    transport.url = "https://mcp.hud.ai/v3/mcp"
    transport.headers = {}
    transport.timeout = 30
    transport.auth = None
    transport.session_id = "test-session"
    transport.protocol_version = None
    transport.request_headers = {}
    transport.sse_read_timeout = 30
    transport._is_initialized_notification = lambda msg: False

    fake_client = _FakeClient(responses)
    write_send, write_recv = anyio.create_memory_object_stream(10)
    read_send, read_recv = anyio.create_memory_object_stream(10)
    original_client_timeout = settings.client_timeout

    try:
        settings.client_timeout = client_timeout_seconds
        result = None
        caught = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                transport.post_writer,
                fake_client,
                write_recv,
                read_send,
                write_send.clone(),
                lambda: None,
                tg,
            )

            try:
                async with ClientSession(read_recv, write_send) as session:
                    import mcp.types as types

                    result = await session.send_request(
                        types.ClientRequest(types.PingRequest()),
                        types.EmptyResult,
                        request_read_timeout_seconds=timedelta(seconds=request_timeout_seconds),
                    )
            except Exception as exc:  # noqa: BLE001
                caught = exc
            finally:
                await write_send.aclose()
                tg.cancel_scope.cancel()

        if caught is not None:
            if isinstance(caught, BaseExceptionGroup) and len(caught.exceptions) == 1:
                raise caught.exceptions[0]
            raise caught

        return result, fake_client.calls
    finally:
        settings.client_timeout = original_client_timeout


class TestJsonResponseReliability:
    """Invalid JSON/read failures should fail the caller promptly, not hang."""

    def test_invalid_json_fails_promptly_without_request_timeout(self):
        import httpx
        from mcp.shared.exceptions import McpError

        with pytest.raises(McpError) as excinfo:
            _run(
                _send_ping_through_transport(
                    [_FakeResponse(body=b"{{invalid json", headers={"content-type": "application/json"})]
                )
            )

        assert excinfo.value.error.code != httpx.codes.REQUEST_TIMEOUT, (
            "Invalid JSON should surface as a transport failure, not a request timeout"
        )

    def test_read_error_fails_promptly_without_request_timeout(self):
        import httpx
        from mcp.shared.exceptions import McpError

        with pytest.raises(McpError) as excinfo:
            _run(
                _send_ping_through_transport(
                    [
                        _FakeResponse(
                            read_error=IOError("Connection reset by peer during response read"),
                            headers={"content-type": "application/json"},
                        )
                    ]
                )
            )

        assert excinfo.value.error.code != httpx.codes.REQUEST_TIMEOUT, (
            "Read errors should surface as a transport failure, not a request timeout"
        )


class TestTimeoutConfiguration:
    """Remote transport should honor a dedicated SSE timeout budget."""

    def test_connect_mcp_uses_dedicated_sse_timeout(self):
        with patch("hud.environment.connectors.mcp_config._build_transport") as mock_build:
            with _patched_runtime_settings(client_timeout=900, sse_read_timeout=37):
                from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

                mock_build.return_value = MagicMock()
                mixin = MCPConfigConnectorMixin.__new__(MCPConfigConnectorMixin)
                mixin._add_connection = MagicMock()
                mixin.connect_mcp({"test": {"url": "https://mcp.hud.ai/v3/mcp"}})

        built_config = mock_build.call_args[0][0]
        assert built_config["sse_read_timeout"] == 37

    def test_zero_sse_timeout_becomes_none(self):
        with patch("hud.environment.connectors.mcp_config._build_transport") as mock_build:
            with _patched_runtime_settings(client_timeout=900, sse_read_timeout=0):
                from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

                mock_build.return_value = MagicMock()
                mixin = MCPConfigConnectorMixin.__new__(MCPConfigConnectorMixin)
                mixin._add_connection = MagicMock()
                mixin.connect_mcp({"test": {"url": "https://mcp.hud.ai/v3/mcp"}})

        built_config = mock_build.call_args[0][0]
        assert built_config.get("sse_read_timeout") is None


class TestTransportRetryOn5xx:
    """Transient gateway failures should recover; client errors should not."""

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    def test_transient_5xx_eventually_succeeds(self, status_code: int):
        result, call_count = _run(
            _send_ping_through_transport(
                [
                    _FakeResponse(status_code=status_code),
                    _success_response(),
                ],
                request_timeout_seconds=1.2,
            )
        )

        assert result is not None
        assert call_count >= 2

    def test_400_fails_without_retrying(self):
        from mcp.shared.exceptions import McpError

        with pytest.raises(McpError):
            _run(
                _send_ping_through_transport(
                    [
                        _FakeResponse(status_code=400),
                        _success_response(),
                    ]
                )
            )

