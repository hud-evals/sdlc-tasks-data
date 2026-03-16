from __future__ import annotations

from types import SimpleNamespace

import pytest

from hud.eval.manager import _enforce_parallel_hosted_containment, _task_uses_hud_hosted_remote


def _make_task(*, hub_config=None, mcp_config=None):
    env = SimpleNamespace(_hub_config=hub_config, _mcp_config=mcp_config)
    return SimpleNamespace(env=env)


def test_grouped_hud_hosted_runs_fail_closed() -> None:
    task = _make_task(hub_config={"name": "browser"})

    with pytest.raises(
        RuntimeError,
        match="Grouped runs against HUD-hosted remote environments are temporarily blocked",
    ) as excinfo:
        _enforce_parallel_hosted_containment([task], 2)

    assert "cross-trace contamination risk" in str(excinfo.value)
    assert "group=1" in str(excinfo.value)


def test_single_trace_hud_hosted_runs_remain_allowed() -> None:
    task = _make_task(hub_config={"name": "browser"})

    _enforce_parallel_hosted_containment([task], 1)


def test_grouped_non_hud_remote_runs_are_not_blocked() -> None:
    task = _make_task(mcp_config={"example": {"url": "https://mcp.example.com"}})

    _enforce_parallel_hosted_containment([task], 2)
    assert _task_uses_hud_hosted_remote(task) is False


def test_grouped_local_runs_are_not_blocked() -> None:
    task = _make_task(mcp_config={"filesystem": {"command": "python", "args": ["-m", "srv"]}})

    _enforce_parallel_hosted_containment([task], 3)
    assert _task_uses_hud_hosted_remote(task) is False


def test_hud_mcp_config_path_is_treated_as_hosted_remote() -> None:
    task = _make_task(mcp_config={"hud": {"url": "https://mcp.hud.ai/browser"}})

    assert _task_uses_hud_hosted_remote(task) is True
