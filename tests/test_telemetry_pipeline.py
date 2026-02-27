"""Tests for the telemetry pipeline — trace propagation, URL construction, and retry logic."""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hud.settings import Settings


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestInstrumentTraceId:
    """Verify the instrument decorator retrieves trace_id without import errors."""

    def test_get_trace_id_no_import_error(self):
        from hud.telemetry.instrument import _get_trace_id

        result = _get_trace_id()
        assert result is None or isinstance(result, str)

    def test_get_trace_id_returns_value_in_context(self):
        from hud.eval.context import set_trace_context
        from hud.telemetry.instrument import _get_trace_id

        with set_trace_context("abc123"):
            result = _get_trace_id()
        assert result == "abc123"

    def test_span_queued_with_trace_id(self):
        from hud.eval.context import set_trace_context
        from hud.telemetry.instrument import instrument

        queued = []

        @instrument
        def sample_func(x: int) -> int:
            return x * 2

        with (
            set_trace_context("trace-for-test"),
            patch("hud.telemetry.instrument.queue_span", side_effect=lambda s: queued.append(s)),
        ):
            result = sample_func(5)

        assert result == 10
        assert len(queued) == 1
        assert queued[0]["trace_id"] is not None
        assert len(queued[0]["trace_id"]) == 32


class TestConnectionTraceExtraction:
    """Verify that Connector.call_tool injects trace_id into MCP metadata."""

    def test_trace_id_injected_in_call(self):
        from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

        connector = Connector(
            transport=MagicMock(),
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.LOCAL,
        )

        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.is_error = False
        mock_result.structuredContent = None
        mock_result.structured_content = None
        mock_result.content = []
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        with patch("hud.eval.context.get_current_trace_id", return_value="test-trace-id"):
            _run(connector.call_tool("my_tool", {"arg": "value"}))

        call_kwargs = mock_client.call_tool.call_args
        all_args_str = str(call_kwargs)
        assert "_hud_trace_id" in all_args_str

    def test_call_without_trace_id_no_crash(self):
        from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

        connector = Connector(
            transport=MagicMock(),
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.LOCAL,
        )

        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.is_error = False
        mock_result.structuredContent = None
        mock_result.structured_content = None
        mock_result.content = []
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        with patch("hud.eval.context.get_current_trace_id", return_value=None):
            result = _run(connector.call_tool("my_tool", {"arg": "value"}))

        assert result is not None

    def test_trace_id_not_injected_when_none(self):
        from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

        connector = Connector(
            transport=MagicMock(),
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.LOCAL,
        )

        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.is_error = False
        mock_result.structuredContent = None
        mock_result.structured_content = None
        mock_result.content = []
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        with patch("hud.eval.context.get_current_trace_id", return_value=None):
            _run(connector.call_tool("my_tool", {}))

        call_args = mock_client.call_tool.call_args
        all_kwargs = call_args.kwargs if call_args.kwargs else {}
        assert "meta" not in all_kwargs
        assert "_meta" not in all_kwargs


class TestTelemetryUrl:
    """Verify URL construction produces no double slashes or trailing slash."""

    def test_no_trailing_slash(self):
        s = Settings(HUD_TELEMETRY_URL="https://telemetry.hud.ai/v3/api")
        assert not s.hud_telemetry_url.endswith("/")

    def test_default_no_trailing_slash(self):
        s = Settings()
        assert not s.hud_telemetry_url.endswith("/")

    def test_upload_url_no_double_slash(self):
        s = Settings()
        url = f"{s.hud_telemetry_url}/trace/spans"
        assert "//" not in url.split("://", 1)[1]

    def test_url_valid_format(self):
        s = Settings()
        assert re.match(r"https?://[^/]+/[^/].*[^/]$", s.hud_telemetry_url)


class TestRequestRetry:
    """Verify that 5xx status codes trigger retries."""

    def test_502_is_retried(self):
        from hud.shared.requests import make_request

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        response_502 = MagicMock(spec=httpx.Response)
        response_502.status_code = 502
        response_502.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "502 Bad Gateway", request=MagicMock(), response=response_502
            )
        )

        response_200 = MagicMock(spec=httpx.Response)
        response_200.status_code = 200
        response_200.raise_for_status = MagicMock()
        response_200.json = MagicMock(return_value={"ok": True})

        mock_client.request = AsyncMock(side_effect=[response_502, response_200])

        result = _run(make_request(
            "GET",
            "https://telemetry.hud.ai/v3/api/trace/spans",
            api_key="test-key",
            max_retries=3,
            retry_delay=0.01,
            client=mock_client,
        ))
        assert result == {"ok": True}
        assert mock_client.request.call_count == 2

    def test_503_is_retried(self):
        from hud.shared.requests import make_request

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        response_503 = MagicMock(spec=httpx.Response)
        response_503.status_code = 503
        response_503.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable", request=MagicMock(), response=response_503
            )
        )

        response_200 = MagicMock(spec=httpx.Response)
        response_200.status_code = 200
        response_200.raise_for_status = MagicMock()
        response_200.json = MagicMock(return_value={"ok": True})

        mock_client.request = AsyncMock(side_effect=[response_503, response_200])

        result = _run(make_request(
            "GET",
            "https://telemetry.hud.ai/v3/api/trace/spans",
            api_key="test-key",
            max_retries=3,
            retry_delay=0.01,
            client=mock_client,
        ))
        assert result == {"ok": True}
        assert mock_client.request.call_count == 2

    def test_400_not_retried(self):
        from hud.shared.exceptions import HudRequestError
        from hud.shared.requests import make_request

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        response_400 = MagicMock(spec=httpx.Response)
        response_400.status_code = 400
        response_400.text = "Bad Request"
        response_400.headers = {}
        response_400.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request", request=MagicMock(), response=response_400
            )
        )

        mock_client.request = AsyncMock(return_value=response_400)

        with pytest.raises(HudRequestError):
            _run(make_request(
                "GET",
                "https://telemetry.hud.ai/v3/api/trace/spans",
                api_key="test-key",
                max_retries=3,
                retry_delay=0.01,
                client=mock_client,
            ))

        assert mock_client.request.call_count == 1
