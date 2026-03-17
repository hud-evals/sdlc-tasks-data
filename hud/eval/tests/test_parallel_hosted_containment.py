from __future__ import annotations

import pytest

from hud.environment import Environment
from hud.eval.manager import _enforce_parallel_hosted_containment, _task_uses_hud_hosted_remote, run_eval
from hud.eval.task import Task


def _make_hosted_task() -> Task:
    return Task(env={"name": "browser"}, args={})


def _make_local_task() -> Task:
    return Task(env=Environment("local-only"), args={})


@pytest.mark.asyncio
async def test_grouped_hud_hosted_runs_fail_closed_at_run_eval_entrypoint() -> None:
    task = _make_hosted_task()

    with pytest.raises(
        RuntimeError,
        match="Grouped runs against HUD-hosted remote environments are temporarily blocked",
    ) as excinfo:
        async with run_eval([task], group=2, quiet=True):
            pytest.fail("Grouped HUD-hosted runs should fail closed before execution starts")

    assert "cross-trace contamination risk" in str(excinfo.value)
    assert "group=1" in str(excinfo.value)


def test_mixed_batches_do_not_trip_global_guard() -> None:
    _enforce_parallel_hosted_containment([_make_hosted_task(), _make_local_task()], 2)


def test_hub_connected_tasks_are_treated_as_hosted_remote() -> None:
    task = _make_hosted_task()

    assert _task_uses_hud_hosted_remote(task) is True


def test_grouped_local_runs_are_not_blocked() -> None:
    task = _make_local_task()

    _enforce_parallel_hosted_containment([task], 3)
    assert _task_uses_hud_hosted_remote(task) is False
