import asyncio

import pytest

from hud.patches import mcp_patches
from mcp.client.streamable_http import StreamableHTTPTransport


class DummyResponse:
    async def aread(self) -> bytes:
        return b"{not-valid-json"


class DummyWriter:
    async def send(self, _message: object) -> None:
        return None


async def _invoke_handle_json_response() -> None:
    mcp_patches.apply_all_patches()

    with pytest.raises(Exception):
        await StreamableHTTPTransport._handle_json_response(
            object(),
            DummyResponse(),
            DummyWriter(),
            is_initialization=False,
        )


def test_handle_json_response_reraises_parse_errors() -> None:
    asyncio.run(_invoke_handle_json_response())
