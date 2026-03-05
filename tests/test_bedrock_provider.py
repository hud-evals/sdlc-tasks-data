"""Tests for AWS Bedrock provider integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage


def _make_beta_message(text: str = "Hello") -> BetaMessage:
    return BetaMessage(
        id="msg-001",
        type="message",
        role="assistant",
        content=[BetaTextBlock(type="text", text=text)],
        model="claude-sonnet-4-5",
        stop_reason="end_turn",
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )


def _make_bedrock_client() -> AsyncAnthropicBedrock:
    return AsyncAnthropicBedrock(
        aws_access_key="test-key",
        aws_secret_key="test-secret",
        aws_region="us-east-1",
    )


def _make_standard_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key="sk-ant-test-key")


class _AsyncIterator:
    """Async iterator that yields nothing (for mocking stream)."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class TestBedrockStreaming:
    """Bedrock client uses create() not stream()."""

    def test_bedrock_client_uses_create(self) -> None:
        from hud.agents.claude import ClaudeAgent

        client = _make_bedrock_client()
        mock_create = AsyncMock(return_value=_make_beta_message())
        client.beta.messages.create = mock_create  # type: ignore[assignment]

        agent = ClaudeAgent(model_client=client, validate_api_key=False)
        agent.claude_tools = []
        agent.tool_mapping = {}
        agent._required_betas = set()

        messages: list = [{"role": "user", "content": "Hi"}]
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(agent.get_response(messages))
        finally:
            loop.close()

        mock_create.assert_called_once()

    def test_standard_client_uses_stream(self) -> None:
        from hud.agents.claude import ClaudeAgent

        client = _make_standard_client()

        mock_message = _make_beta_message()

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = MagicMock(return_value=_AsyncIterator())
        mock_stream.get_final_message = AsyncMock(return_value=mock_message)

        client.beta.messages.stream = MagicMock(return_value=mock_stream)  # type: ignore[assignment]

        agent = ClaudeAgent(model_client=client, validate_api_key=False)
        agent.claude_tools = []
        agent.tool_mapping = {}
        agent._required_betas = set()

        messages: list = [{"role": "user", "content": "Hi"}]
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(agent.get_response(messages))
        finally:
            loop.close()

        client.beta.messages.stream.assert_called_once()

    def test_bedrock_missing_boto3_raises(self) -> None:
        from hud.agents.claude import ClaudeAgent

        client = _make_bedrock_client()
        mock_create = AsyncMock(side_effect=ModuleNotFoundError("No module named 'boto3'"))
        client.beta.messages.create = mock_create  # type: ignore[assignment]

        agent = ClaudeAgent(model_client=client, validate_api_key=False)
        agent.claude_tools = []
        agent.tool_mapping = {}
        agent._required_betas = set()

        messages: list = [{"role": "user", "content": "Hi"}]
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(Exception) as exc_info:
                loop.run_until_complete(agent.get_response(messages))
            error_msg = str(exc_info.value).lower()
            assert "boto3" in error_msg or "bedrock" in error_msg
        finally:
            loop.close()

    def test_bedrock_response_appended_to_messages(self) -> None:
        from hud.agents.claude import ClaudeAgent

        client = _make_bedrock_client()
        mock_create = AsyncMock(return_value=_make_beta_message("Bedrock response"))
        client.beta.messages.create = mock_create  # type: ignore[assignment]

        agent = ClaudeAgent(model_client=client, validate_api_key=False)
        agent.claude_tools = []
        agent.tool_mapping = {}
        agent._required_betas = set()

        messages: list = [{"role": "user", "content": "Hi"}]
        initial_len = len(messages)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(agent.get_response(messages))
        finally:
            loop.close()

        assert len(messages) > initial_len
        assert messages[-1]["role"] == "assistant"


class TestBedrockClientInit:
    """ClaudeAgent accepts Bedrock clients correctly."""

    def test_accepts_bedrock_client(self) -> None:
        from hud.agents.claude import ClaudeAgent

        client = _make_bedrock_client()
        agent = ClaudeAgent(model_client=client, validate_api_key=False)
        assert agent.anthropic_client is client

    def test_bedrock_client_skips_api_key_validation(self) -> None:
        from hud.agents.claude import ClaudeAgent

        client = _make_bedrock_client()
        agent = ClaudeAgent(model_client=client, validate_api_key=True)
        assert agent.anthropic_client is client

    def test_config_model_client_type_accepts_bedrock(self) -> None:
        from hud.agents.types import ClaudeConfig

        client = _make_bedrock_client()
        config = ClaudeConfig(model_client=client)
        assert config.model_client is client


class TestBedrockARNDetection:
    """CLI detects Bedrock ARNs and wires credentials."""

    def test_valid_arn_triggers_bedrock_detection(self) -> None:
        """Various Bedrock ARN formats should produce a Bedrock client via get_agent_kwargs."""
        from hud.cli.eval import EvalConfig
        from hud.types import AgentType

        valid_arns = [
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "arn:aws:bedrock:eu-west-1:987654321098:inference-profile/eu.anthropic.claude-v2",
            "arn:aws:bedrock:ap-southeast-1:111222333444:inference-profile/custom-profile",
        ]

        mock_settings = MagicMock()
        mock_settings.aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
        mock_settings.aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_settings.aws_region = "us-east-1"
        mock_settings.api_key = "hud-test-key"
        mock_settings.anthropic_api_key = "sk-ant-test"
        mock_settings.hud_gateway_url = "https://inference.hud.ai"

        for arn in valid_arns:
            with patch("hud.cli.eval.settings", mock_settings), \
                 patch("hud.settings.settings", mock_settings):
                config = EvalConfig(
                    agent_type=AgentType.CLAUDE,
                    model=arn,
                )
                kwargs = config.get_agent_kwargs()
            assert isinstance(kwargs.get("model_client"), AsyncAnthropicBedrock), \
                f"Expected Bedrock client for ARN: {arn}"

    def test_normal_model_does_not_trigger_bedrock(self) -> None:
        """Normal model names should NOT create a Bedrock client."""
        from hud.cli.eval import EvalConfig
        from hud.types import AgentType

        normal_models = [
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "gpt-4o",
        ]

        mock_settings = MagicMock()
        mock_settings.api_key = "hud-test-key"
        mock_settings.anthropic_api_key = "sk-ant-test"
        mock_settings.hud_gateway_url = "https://inference.hud.ai"

        for model in normal_models:
            with patch("hud.cli.eval.settings", mock_settings), \
                 patch("hud.settings.settings", mock_settings):
                config = EvalConfig(
                    agent_type=AgentType.CLAUDE,
                    model=model,
                )
                kwargs = config.get_agent_kwargs()
            assert not isinstance(kwargs.get("model_client"), AsyncAnthropicBedrock), \
                f"Expected no Bedrock client for model: {model}"

    def test_bedrock_arn_creates_bedrock_client(self) -> None:
        from hud.cli.eval import EvalConfig
        from hud.types import AgentType

        bedrock_arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

        mock_settings = MagicMock()
        mock_settings.aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
        mock_settings.aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_settings.aws_region = "us-east-1"
        mock_settings.api_key = "hud-test-key"
        mock_settings.anthropic_api_key = "sk-ant-test"
        mock_settings.hud_gateway_url = "https://inference.hud.ai"

        with patch("hud.cli.eval.settings", mock_settings), \
             patch("hud.settings.settings", mock_settings):
            config = EvalConfig(
                agent_type=AgentType.CLAUDE,
                model=bedrock_arn,
            )
            kwargs = config.get_agent_kwargs()

        assert "model_client" in kwargs
        assert isinstance(kwargs["model_client"], AsyncAnthropicBedrock)
