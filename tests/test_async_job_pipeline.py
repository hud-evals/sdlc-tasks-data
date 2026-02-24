"""Integration tests for async job pipeline and taskset propagation.

Verifies that _send_job_enter is fully async, that the --taskset parameter
propagates end-to-end from CLI through run_dataset to hud.eval, that
JobEnterPayload includes the taskset field, and that response parsing
handles missing keys gracefully.
"""
from __future__ import annotations

import asyncio
import ast
import inspect
from unittest.mock import AsyncMock, MagicMock, patch


class TestSendJobEnterAsync:
    """_send_job_enter must be a proper async function using async HTTP."""

    def test_is_coroutine_function(self):
        from hud.eval.manager import _send_job_enter
        assert inspect.iscoroutinefunction(_send_job_enter), (
            "_send_job_enter must be an async function (use 'async def')"
        )

    def test_no_sync_httpx_post(self):
        """Must not use synchronous httpx.post() inside the function."""
        with open("hud/eval/manager.py") as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "_send_job_enter":
                    func_source = ast.get_source_segment(source, node)
                    if func_source and "httpx.post(" in func_source:
                        assert False, (
                            "_send_job_enter uses synchronous httpx.post(). "
                            "Must use async HTTP (e.g. httpx.AsyncClient) "
                            "to avoid blocking the event loop."
                        )


class TestResponseParsing:
    """_send_job_enter must handle missing task_version_ids gracefully."""

    def test_handles_missing_task_version_ids(self):
        """Response without task_version_ids should return None, not crash."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"status": "ok"}

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client):
                with patch("hud.settings.settings") as mock_settings:
                    mock_settings.telemetry_enabled = True
                    mock_settings.api_key = "test-key"
                    mock_settings.hud_api_url = "https://api.test"

                    result = await _send_job_enter(
                        job_id="j1", name="test",
                        variants=None, group=1, api_key="test-key",
                        taskset="my-taskset",
                    )
                    return result

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert result is None or isinstance(result, list)

    def test_returns_ids_when_present(self):
        """When API returns task_version_ids, function should return them."""
        from hud.eval.manager import _send_job_enter

        async def _run():
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "task_version_ids": ["tv-1", "tv-2"]
            }

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client):
                with patch("hud.settings.settings") as mock_settings:
                    mock_settings.telemetry_enabled = True
                    mock_settings.api_key = "test-key"
                    mock_settings.hud_api_url = "https://api.test"

                    return await _send_job_enter(
                        job_id="j1", name="test",
                        variants=None, group=1, api_key="test-key",
                        taskset="my-taskset",
                    )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert result is not None, "Should return task_version_ids when present"
        assert isinstance(result, list)
        assert len(result) == 2


class TestTasksetPlumbing:
    """--taskset must propagate from CLI config through to the eval manager."""

    def test_run_dataset_accepts_taskset(self):
        from hud.datasets.runner import run_dataset
        sig = inspect.signature(run_dataset)
        assert "taskset" in sig.parameters, (
            "run_dataset() must accept a 'taskset' parameter"
        )

    def test_job_enter_payload_has_taskset(self):
        from hud.eval.types import JobEnterPayload
        fields = JobEnterPayload.model_fields
        assert "taskset" in fields, (
            "JobEnterPayload must have a 'taskset' field"
        )

    def test_job_enter_payload_taskset_serializes(self):
        from hud.eval.types import JobEnterPayload
        payload = JobEnterPayload(name="test", taskset="my-taskset")
        data = payload.model_dump(exclude_none=True)
        assert data.get("taskset") == "my-taskset", (
            "JobEnterPayload with taskset='my-taskset' must serialize "
            "the taskset field in model_dump()."
        )

    def test_send_job_enter_accepts_taskset(self):
        from hud.eval.manager import _send_job_enter
        sig = inspect.signature(_send_job_enter)
        assert "taskset" in sig.parameters, (
            "_send_job_enter must accept a 'taskset' parameter"
        )

    def test_cli_passes_taskset_to_run_dataset(self):
        """_run_evaluation must pass taskset to run_dataset()."""
        with open("hud/cli/eval.py") as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "_run_evaluation":
                    func_source = ast.get_source_segment(source, node)
                    if func_source:
                        fn_tree = ast.parse(func_source)
                        for n in ast.walk(fn_tree):
                            if not isinstance(n, ast.Call):
                                continue
                            func = n.func
                            if isinstance(func, ast.Name) and func.id == "run_dataset":
                                for kw in n.keywords:
                                    if kw.arg == "taskset":
                                        return
                        assert False, (
                            "_run_evaluation() calls run_dataset() without "
                            "passing taskset=. The --taskset CLI option is "
                            "captured but never forwarded."
                        )

    def test_run_dataset_forwards_taskset(self):
        """run_dataset must pass taskset= to hud.eval()."""
        with open("hud/datasets/runner.py") as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "run_dataset":
                    func_source = ast.get_source_segment(source, node)
                    if func_source:
                        assert "taskset" in func_source, (
                            "run_dataset() body does not reference 'taskset'. "
                            "The taskset parameter must be forwarded."
                        )
                        return
        assert False, "run_dataset function not found"
