"""Tests for reasoning/thinking content separation across all agent types.

Each agent must route provider-specific reasoning content to AgentResponse.reasoning
and keep AgentResponse.content clean (no "Thinking:" prefix or reasoning text).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _MockStream:
    """Async context manager + async iterable that yields nothing and returns a final message."""

    def __init__(self, final_message: Any) -> None:
        self._final_message = final_message

    async def __aenter__(self) -> "_MockStream":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def __aiter__(self) -> "_MockStream":
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return self._final_message


# ---------------------------------------------------------------------------
# Claude Agent
# ---------------------------------------------------------------------------


class TestClaudeReasoningSeparation:
    """Claude agent separates thinking blocks from content."""

    @pytest.mark.asyncio
    async def test_thinking_goes_to_reasoning_field(self) -> None:
        from hud.agents.claude import ClaudeAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            agent = ClaudeAgent.create(model_client=client, validate_api_key=False)
            agent.claude_tools = []
            agent.tool_mapping = {}
            agent.has_computer_tool = False
            agent._required_betas = set()
            agent._initialized = True

            mock_resp = MagicMock()
            thinking = MagicMock()
            thinking.type = "thinking"
            thinking.thinking = "Deep analysis of the problem"
            text = MagicMock()
            text.type = "text"
            text.text = "Final answer"
            mock_resp.content = [thinking, text]

            client.beta.messages.stream = MagicMock(return_value=_MockStream(mock_resp))

            response = await agent.get_response([{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])

            assert response.reasoning is not None
            assert "Deep analysis" in response.reasoning

    @pytest.mark.asyncio
    async def test_content_is_clean(self) -> None:
        from hud.agents.claude import ClaudeAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            agent = ClaudeAgent.create(model_client=client, validate_api_key=False)
            agent.claude_tools = []
            agent.tool_mapping = {}
            agent.has_computer_tool = False
            agent._required_betas = set()
            agent._initialized = True

            mock_resp = MagicMock()
            thinking = MagicMock()
            thinking.type = "thinking"
            thinking.thinking = "Internal reasoning about the task"
            text = MagicMock()
            text.type = "text"
            text.text = "Visible answer"
            mock_resp.content = [thinking, text]

            client.beta.messages.stream = MagicMock(return_value=_MockStream(mock_resp))

            response = await agent.get_response([{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])

            assert "Thinking:" not in response.content
            assert "Internal reasoning" not in response.content
            assert response.content == "Visible answer"

    @pytest.mark.asyncio
    async def test_text_only_response_has_no_reasoning(self) -> None:
        from hud.agents.claude import ClaudeAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            agent = ClaudeAgent.create(model_client=client, validate_api_key=False)
            agent.claude_tools = []
            agent.tool_mapping = {}
            agent.has_computer_tool = False
            agent._required_betas = set()
            agent._initialized = True

            mock_resp = MagicMock()
            text = MagicMock()
            text.type = "text"
            text.text = "Just a text reply"
            mock_resp.content = [text]

            client.beta.messages.stream = MagicMock(return_value=_MockStream(mock_resp))

            response = await agent.get_response([{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])

            assert response.reasoning is None


# ---------------------------------------------------------------------------
# Gemini Agent
# ---------------------------------------------------------------------------


class TestGeminiReasoningSeparation:
    """Gemini agent separates thought parts from content."""

    @pytest.mark.asyncio
    async def test_thought_goes_to_reasoning_field(self) -> None:
        from hud.agents.gemini import GeminiAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            client.aio = MagicMock()
            client.aio.models = MagicMock()
            agent = GeminiAgent.create(model_client=client, validate_api_key=False)
            agent.gemini_tools = []
            agent._initialized = True

            mock_resp = MagicMock()
            candidate = MagicMock()

            thought_part = MagicMock()
            thought_part.text = "Reasoning through options"
            thought_part.function_call = None
            thought_part.thought = True

            text_part = MagicMock()
            text_part.text = "The answer is 42"
            text_part.function_call = None
            text_part.thought = False

            candidate.content = MagicMock()
            candidate.content.parts = [thought_part, text_part]
            mock_resp.candidates = [candidate]

            client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

            response = await agent.get_response([])

            assert response.reasoning is not None
            assert "Reasoning through options" in response.reasoning

    @pytest.mark.asyncio
    async def test_content_is_clean(self) -> None:
        from hud.agents.gemini import GeminiAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            client.aio = MagicMock()
            client.aio.models = MagicMock()
            agent = GeminiAgent.create(model_client=client, validate_api_key=False)
            agent.gemini_tools = []
            agent._initialized = True

            mock_resp = MagicMock()
            candidate = MagicMock()

            thought_part = MagicMock()
            thought_part.text = "Internal thought process"
            thought_part.function_call = None
            thought_part.thought = True

            text_part = MagicMock()
            text_part.text = "Clean response"
            text_part.function_call = None
            text_part.thought = False

            candidate.content = MagicMock()
            candidate.content.parts = [thought_part, text_part]
            mock_resp.candidates = [candidate]

            client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

            response = await agent.get_response([])

            assert "Thinking:" not in response.content
            assert "Internal thought process" not in response.content
            assert response.content == "Clean response"

    @pytest.mark.asyncio
    async def test_thought_part_not_classified_as_text(self) -> None:
        """A thought part with text content must go to reasoning, not content.

        This tests the condition-order fix: thought check must come before text check.
        """
        from hud.agents.gemini import GeminiAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            client.aio = MagicMock()
            client.aio.models = MagicMock()
            agent = GeminiAgent.create(model_client=client, validate_api_key=False)
            agent.gemini_tools = []
            agent._initialized = True

            mock_resp = MagicMock()
            candidate = MagicMock()

            # A thought part that also has .text set — the condition-order bug
            # causes this to be classified as regular text
            thought_part = MagicMock()
            thought_part.text = "This is thought content with text"
            thought_part.function_call = None
            thought_part.thought = True

            text_part = MagicMock()
            text_part.text = "Actual response"
            text_part.function_call = None
            text_part.thought = False

            candidate.content = MagicMock()
            candidate.content.parts = [thought_part, text_part]
            mock_resp.candidates = [candidate]

            client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

            response = await agent.get_response([])

            assert "This is thought content with text" not in response.content
            assert response.content == "Actual response"
            assert response.reasoning is not None
            assert "This is thought content with text" in response.reasoning


# ---------------------------------------------------------------------------
# OpenAI Responses API Agent
# ---------------------------------------------------------------------------


class TestOpenAIReasoningSeparation:
    """OpenAI Responses API agent separates reasoning items from content."""

    @pytest.mark.asyncio
    async def test_reasoning_goes_to_reasoning_field(self) -> None:
        from hud.agents.openai import OpenAIAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            client.responses = MagicMock()
            agent = OpenAIAgent.create(model_client=client, validate_api_key=False)
            agent._openai_tools = []
            agent._tool_name_map = {}
            agent._initialized = True

            mock_resp = MagicMock()
            mock_resp.id = "resp_123"

            reasoning_item = MagicMock()
            reasoning_item.type = "reasoning"
            summary = MagicMock()
            summary.text = "Let me think about this carefully"
            reasoning_item.summary = [summary]

            message_item = MagicMock()
            message_item.type = "message"
            from openai.types.responses import ResponseOutputText as ROT
            output_text = MagicMock(spec=ROT)
            output_text.text = "Here is my final answer"
            output_text.__class__ = ROT
            message_item.content = [output_text]

            mock_resp.output = [reasoning_item, message_item]
            client.responses.create = AsyncMock(return_value=mock_resp)

            response = await agent.get_response([])

            assert response.reasoning is not None
            assert "Let me think about this carefully" in response.reasoning

    @pytest.mark.asyncio
    async def test_content_is_clean(self) -> None:
        from hud.agents.openai import OpenAIAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            client.responses = MagicMock()
            agent = OpenAIAgent.create(model_client=client, validate_api_key=False)
            agent._openai_tools = []
            agent._tool_name_map = {}
            agent._initialized = True

            mock_resp = MagicMock()
            mock_resp.id = "resp_456"

            reasoning_item = MagicMock()
            reasoning_item.type = "reasoning"
            summary = MagicMock()
            summary.text = "Internal reasoning step"
            reasoning_item.summary = [summary]

            message_item = MagicMock()
            message_item.type = "message"
            from openai.types.responses import ResponseOutputText as ROT
            output_text = MagicMock(spec=ROT)
            output_text.text = "Clean answer"
            output_text.__class__ = ROT

            message_item.content = [output_text]
            mock_resp.output = [reasoning_item, message_item]
            client.responses.create = AsyncMock(return_value=mock_resp)

            response = await agent.get_response([])

            assert "Thinking:" not in response.content
            assert "Internal reasoning step" not in response.content
            assert response.content == "Clean answer"

    @pytest.mark.asyncio
    async def test_text_only_response_has_no_reasoning(self) -> None:
        from hud.agents.openai import OpenAIAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            client = MagicMock()
            client.responses = MagicMock()
            agent = OpenAIAgent.create(model_client=client, validate_api_key=False)
            agent._openai_tools = []
            agent._tool_name_map = {}
            agent._initialized = True

            mock_resp = MagicMock()
            mock_resp.id = "resp_789"

            message_item = MagicMock()
            message_item.type = "message"
            from openai.types.responses import ResponseOutputText as ROT
            output_text = MagicMock(spec=ROT)
            output_text.text = "Simple text"
            output_text.__class__ = ROT

            message_item.content = [output_text]
            mock_resp.output = [message_item]
            client.responses.create = AsyncMock(return_value=mock_resp)

            response = await agent.get_response([])

            assert response.reasoning is None


# ---------------------------------------------------------------------------
# Grounded OpenAI Agent
# ---------------------------------------------------------------------------


class TestGroundedOpenAIReasoningField:
    """GroundedOpenAI agent populates reasoning from message.reasoning_content."""

    @pytest.mark.asyncio
    async def test_reasoning_populated(self) -> None:
        from openai import AsyncOpenAI as _AsyncOpenAI

        from hud.agents.grounded_openai import GroundedOpenAIChatAgent
        from hud.tools.grounding import GrounderConfig

        with patch("hud.settings.settings.telemetry_enabled", False):
            grounder_cfg = GrounderConfig(api_base="http://example", model="qwen")
            fake_openai = _AsyncOpenAI(api_key="test")
            agent = GroundedOpenAIChatAgent.create(
                grounder_config=grounder_cfg,
                openai_client=fake_openai,
                model="gpt-4o",
                initial_screenshot=False,
            )
            agent._initialized = True

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = "Visible answer"
            mock_msg.reasoning_content = "Step by step reasoning here"
            mock_msg.tool_calls = None
            mock_choice.message = mock_msg
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]

            agent.oai.chat.completions.create = AsyncMock(return_value=mock_response)

            png_b64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB"
                "J2n0mQAAAABJRU5ErkJggg=="
            )
            agent.conversation_history = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                    ],
                }
            ]

            response = await agent.get_response(agent.conversation_history)

            assert response.reasoning is not None
            assert "Step by step reasoning" in response.reasoning
            assert response.content == "Visible answer"

    @pytest.mark.asyncio
    async def test_reasoning_on_tool_call(self) -> None:
        from openai import AsyncOpenAI as _AsyncOpenAI

        from hud.agents.grounded_openai import GroundedOpenAIChatAgent
        from hud.tools.grounding import GrounderConfig

        with patch("hud.settings.settings.telemetry_enabled", False):
            grounder_cfg = GrounderConfig(api_base="http://example", model="qwen")
            fake_openai = _AsyncOpenAI(api_key="test")
            agent = GroundedOpenAIChatAgent.create(
                grounder_config=grounder_cfg,
                openai_client=fake_openai,
                model="gpt-4o",
                initial_screenshot=False,
            )
            agent._initialized = True

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = ""
            mock_msg.reasoning_content = "Planning computer action"

            mock_tc = MagicMock()
            mock_tc.id = "tc_001"
            mock_tc.function.name = "computer"
            mock_tc.function.arguments = '{"action": "click", "element_description": "button"}'
            mock_msg.tool_calls = [mock_tc]

            mock_choice.message = mock_msg
            mock_choice.finish_reason = "tool_calls"
            mock_response.choices = [mock_choice]

            agent.oai.chat.completions.create = AsyncMock(return_value=mock_response)

            png_b64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB"
                "J2n0mQAAAABJRU5ErkJggg=="
            )
            agent.conversation_history = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                    ],
                }
            ]

            response = await agent.get_response(agent.conversation_history)

            assert response.reasoning is not None
            assert "Planning computer action" in response.reasoning


# ---------------------------------------------------------------------------
# OpenAI Chat Agent
# ---------------------------------------------------------------------------


class TestOpenAIChatReasoningField:
    """OpenAI Chat agent populates reasoning from message.reasoning_content."""

    @pytest.mark.asyncio
    async def test_reasoning_populated(self) -> None:
        from openai import AsyncOpenAI as _AsyncOpenAI

        from hud.agents.openai_chat import OpenAIChatAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            fake_openai = _AsyncOpenAI(api_key="test")
            agent = OpenAIChatAgent.create(
                openai_client=fake_openai,
                model="gpt-4o",
            )
            agent._initialized = True

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = "Answer text"
            mock_msg.reasoning_content = "Internal reasoning"
            mock_msg.tool_calls = None
            mock_choice.message = mock_msg
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]

            agent.oai.chat.completions.create = AsyncMock(return_value=mock_response)

            response = await agent.get_response([{"role": "user", "content": "Hi"}])

            assert response.reasoning is not None
            assert "Internal reasoning" in response.reasoning

    @pytest.mark.asyncio
    async def test_reasoning_with_tool_calls(self) -> None:
        from openai import AsyncOpenAI as _AsyncOpenAI

        from hud.agents.openai_chat import OpenAIChatAgent

        with patch("hud.settings.settings.telemetry_enabled", False):
            fake_openai = _AsyncOpenAI(api_key="test")
            agent = OpenAIChatAgent.create(
                openai_client=fake_openai,
                model="gpt-4o",
            )
            agent._initialized = True
            agent.mcp_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run bash",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = None
            mock_msg.reasoning_content = "Deciding to run bash command"

            mock_tc = MagicMock()
            mock_tc.id = "call_abc"
            mock_tc.function.name = "bash"
            mock_tc.function.arguments = '{"command": "ls"}'
            mock_msg.tool_calls = [mock_tc]

            mock_choice.message = mock_msg
            mock_choice.finish_reason = "tool_calls"
            mock_response.choices = [mock_choice]

            agent.oai.chat.completions.create = AsyncMock(return_value=mock_response)

            response = await agent.get_response([{"role": "user", "content": "List files"}])

            assert response.reasoning is not None
            assert "Deciding to run bash command" in response.reasoning
