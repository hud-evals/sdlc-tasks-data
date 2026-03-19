from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.environment import Environment
from hud.eval.context import EvalContext


def test_incident_artifact_pack_exists() -> None:
    incident_dir = Path("ops/incidents/INC-2026-03-diagnostics-safe-mode")
    expected = {
        "control-sample.md",
        "incident-sample-a.md",
        "incident-sample-b.md",
        "incident-sample-c.md",
        "incident-sample-d.md",
        "release-window.md",
        "operator-handoff.md",
    }
    assert incident_dir.exists()
    assert expected.issubset({path.name for path in incident_dir.iterdir()})


def _joined_text(result_blocks: list[object]) -> str:
    return "\n".join(getattr(block, "text", "") for block in result_blocks).strip()


@pytest.mark.asyncio
async def test_missing_parent_trace_emits_safe_mode_fallback() -> None:
    """Missing nested trace context should leave a usable degraded parent artifact."""
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
        mock_agent.run = AsyncMock(
            return_value=MagicMock(
                content="downstream probe found a stale handoff state",
                isError=False,
                info={},
            )
        )
        mock_create_agent.return_value = mock_agent

        result = await env._env_call_tool("investigate", {"issue": "fulfillment gap"})

    text = _joined_text(result)
    assert text
    assert any(term in text.lower() for term in ("safe mode", "degraded", "fallback"))
    assert "stale handoff state" in text
    assert mock_run_eval.call_args.kwargs["trace"] is False


@pytest.mark.asyncio
async def test_failed_nested_runs_leave_actionable_safe_mode_context() -> None:
    """Reduced mode should still return actionable failure context to the parent run."""
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
        mock_agent.run = AsyncMock(
            return_value=MagicMock(
                content="Agent failed with error: nested dependency unavailable",
                isError=True,
                info={"error": "nested dependency unavailable"},
            )
        )
        mock_create_agent.return_value = mock_agent

        result = await env._env_call_tool("investigate", {"issue": "billing spike"})

    text = _joined_text(result)
    assert text
    assert any(term in text.lower() for term in ("safe mode", "degraded", "fallback"))
    assert "nested dependency unavailable" in text
    assert mock_run_eval.call_args.kwargs["trace"] is False


@pytest.mark.asyncio
async def test_attached_nested_happy_path_remains_unchanged() -> None:
    """When parent trace context exists, nested diagnostics should stay attached."""
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
        mock_agent.run = AsyncMock(
            return_value=MagicMock(content="nested diagnostics output", isError=False, info={})
        )
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
            result = await env._env_call_tool("investigate", {"issue": "order decline"})
        finally:
            request_ctx.reset(token)

    text = _joined_text(result)
    assert text == "nested diagnostics output"
    assert "safe mode" not in text.lower()
    assert mock_run_eval.call_args.kwargs["trace_id"] == trace_id


@pytest.mark.asyncio
async def test_setup_time_failures_register_visibility_before_connect() -> None:
    """Setup failures should still register visibility before environment connect."""
    order: list[str] = []
    ctx = EvalContext(name="diagnostics-safe-mode", quiet=True)

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
