from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.agents.tests.test_base import MockEvalContext, MockMCPAgent
from hud.environment import Environment
from hud.eval.context import EvalContext
from hud.types import AgentResponse, MCPToolCall


def test_incident_artifact_pack_exists() -> None:
    incident_dir = Path("ops/incidents/INC-2026-03-diagnostics-blackhole")
    expected = {
        "control-sample.md",
        "incident-sample-a.md",
        "incident-sample-b.md",
        "incident-sample-c.md",
        "release-window.md",
        "operator-handoff.md",
    }
    assert incident_dir.exists()
    assert expected.issubset({path.name for path in incident_dir.iterdir()})


@pytest.mark.asyncio
async def test_nested_diagnostics_inherit_parent_lineage() -> None:
    """Nested AgentTool paths should inherit parent lineage from request context."""
    from mcp.server.lowlevel.server import request_ctx
    from mcp.shared.context import RequestContext
    from mcp.types import RequestParams

    from hud.tools import AgentTool

    env = Environment("test")

    @env.scenario()
    async def investigate(issue: str):
        yield {"task": f"Investigate {issue}"}

    agent_tool = AgentTool(env("investigate"), model="claude", trace=True)
    env.add_tool(agent_tool.mcp)
    await env._build_routing()

    with (
        patch("hud.eval.manager.run_eval") as mock_run_eval,
        patch("hud.agents.create_agent") as mock_create_agent,
    ):
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_run_eval.return_value = mock_ctx

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(content="nested diagnostics output"))
        mock_create_agent.return_value = mock_agent

        trace_id = "0123456789abcdef0123456789abcdef"
        req_meta = RequestParams.Meta.model_validate({"_hud_trace_id": trace_id})
        req_context = RequestContext(
            request_id="test-req",
            meta=req_meta,
            session=MagicMock(),
            lifespan_context=None,
        )
        token = request_ctx.set(req_context)  # type: ignore[arg-type]
        try:
            result = await env._env_call_tool("investigate", {"issue": "fulfillment gap"})
        finally:
            request_ctx.reset(token)

    assert len(result) == 1
    assert mock_run_eval.call_args.kwargs["trace_id"] == trace_id


@pytest.mark.asyncio
async def test_failed_nested_runs_populate_ctx_error() -> None:
    """Failed nested execution should surface error context at the run level."""

    class FailOnSecondStepAgent(MockMCPAgent):
        def __init__(self) -> None:
            super().__init__()
            self._step_count = 0

        async def get_response(self, messages: list[dict[str, object]]) -> AgentResponse:
            self._step_count += 1
            if self._step_count == 1:
                return AgentResponse(
                    content="",
                    tool_calls=[MCPToolCall(name="test_tool", arguments={})],
                    done=False,
                )
            raise ValueError("Nested diagnostics step failed")

    ctx = MockEvalContext(prompt="Run diagnostics")
    agent = FailOnSecondStepAgent()

    result = await agent.run(ctx)

    assert result.isError is True
    assert ctx.error is not None
    assert "Nested diagnostics step failed" in str(ctx.error)


@pytest.mark.asyncio
async def test_setup_time_failures_register_visibility_before_connect() -> None:
    """Setup failures should still register visibility before environment connect."""
    order: list[str] = []
    ctx = EvalContext(name="diagnostics-blackhole", quiet=True)

    async def fake_eval_enter() -> None:
        order.append("trace_enter")

    async def fake_eval_exit(error_message: str | None = None) -> None:
        order.append(f"trace_exit:{error_message}")

    async def fake_env_enter(self: Environment) -> Environment:
        order.append("env_enter")
        raise ConnectionError("boom during setup")

    with (
        patch.object(ctx, "_eval_enter", AsyncMock(side_effect=fake_eval_enter)),
        patch.object(ctx, "_eval_exit", AsyncMock(side_effect=fake_eval_exit)),
        patch.object(Environment, "__aenter__", fake_env_enter),
        patch.object(Environment, "__aexit__", AsyncMock(return_value=None)),
    ):
        with pytest.raises(ConnectionError, match="boom during setup"):
            await ctx.__aenter__()

    assert order[0] == "trace_enter"
    assert "env_enter" in order
    assert any(item.startswith("trace_exit:") for item in order)
