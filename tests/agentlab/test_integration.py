from __future__ import annotations

from pathlib import Path

from agentlab.config import AgentLabConfig
from agentlab.evaluation import get_task
from agentlab.policies import ScriptedPolicy
from agentlab.policies.minimind_policy import MiniMindPolicy
from agentlab.runtime import AgentRuntime


def test_calculator_end_to_end_does_not_modify_benchmark(tmp_path):
    task = get_task("calculator_fix_divide_zero")
    source = Path(task.repo_path) / "src" / "calculator" / "core.py"
    before = source.read_bytes()
    result = AgentRuntime(
        ScriptedPolicy(), AgentLabConfig(runs_dir=tmp_path / "runs")
    ).run(task)
    assert result.evaluation.success
    assert result.evaluation.test_passed
    assert result.trajectory.modified_files == ["src/calculator/core.py"]
    assert "raise ValueError" in Path(result.trajectory.trace_path).with_name(
        "final_diff.patch"
    ).read_text(encoding="utf-8")
    assert source.read_bytes() == before


def test_minimind_tool_call_parser():
    action = MiniMindPolicy.parse_response(
        '<think>brief</think><tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
    )
    assert action.kind == "tool"
    assert action.tool_call.name == "read_file"
    assert action.tool_call.arguments == {"path": "a.py"}
    final = MiniMindPolicy.parse_response("<think>x</think>Finished")
    assert final.kind == "final" and final.final_answer == "Finished"


def test_fastapi_module_imports_when_dependency_available():
    import pytest

    pytest.importorskip("fastapi")
    from agentlab.service.app import app

    assert app.title == "MiniMind-AgentLab"
