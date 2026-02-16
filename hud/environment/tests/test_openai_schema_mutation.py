"""Hidden tests for OpenAI schema mutation bug.

Verifies that calling as_openai_chat_tools(strict=True) multiple times
does not corrupt the underlying tool schemas through shared nested references.
"""

from __future__ import annotations

import copy
from typing import Any

import mcp.types as mcp_types

from hud.environment.integrations.openai import OpenAIMixin


def _make_tool(name: str, schema: dict) -> mcp_types.Tool:
    return mcp_types.Tool(name=name, description=f"{name} tool", inputSchema=schema)


def _make_env(tools: list[mcp_types.Tool]):
    class Env(OpenAIMixin):
        def as_tools(self):
            return tools

        async def call_tool(self, name, arguments):
            pass

    return Env()


NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "options": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
        },
    },
}


class TestSchemaNotMutated:

    def test_strict_does_not_mutate_nested_schema(self):
        schema = copy.deepcopy(NESTED_SCHEMA)
        original = copy.deepcopy(schema)
        env = _make_env([_make_tool("search", schema)])

        env.as_openai_chat_tools(strict=True)

        assert schema == original, (
            f"Nested schema was mutated by strict mode. "
            f"Expected: {original}, Got: {schema}"
        )

    def test_second_strict_call_same_result(self):
        schema = copy.deepcopy(NESTED_SCHEMA)
        env = _make_env([_make_tool("search", schema)])

        r1 = env.as_openai_chat_tools(strict=True)
        r2 = env.as_openai_chat_tools(strict=True)

        p1 = r1[0]["function"]["parameters"]
        p2 = r2[0]["function"]["parameters"]
        assert p1 == p2, (
            f"Second strict call returned different result. "
            f"First: {p1}, Second: {p2}"
        )

    def test_responses_tools_does_not_mutate(self):
        schema = copy.deepcopy(NESTED_SCHEMA)
        original = copy.deepcopy(schema)
        env = _make_env([_make_tool("fetch", schema)])

        env.as_openai_responses_tools(strict=True)

        assert schema == original, (
            f"Schema mutated by as_openai_responses_tools(strict=True). "
            f"Expected: {original}, Got: {schema}"
        )

    def test_multiple_tools_independent_after_strict(self):
        schema_a = copy.deepcopy(NESTED_SCHEMA)
        schema_b = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                    },
                },
            },
        }
        orig_a = copy.deepcopy(schema_a)
        orig_b = copy.deepcopy(schema_b)

        env = _make_env([_make_tool("a", schema_a), _make_tool("b", schema_b)])
        env.as_openai_chat_tools(strict=True)

        assert schema_a == orig_a, f"schema_a mutated: {schema_a}"
        assert schema_b == orig_b, f"schema_b mutated: {schema_b}"
