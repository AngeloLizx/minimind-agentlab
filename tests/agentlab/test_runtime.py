from __future__ import annotations

from pathlib import Path

from agentlab.config import AgentLabConfig
from agentlab.evaluation import get_task
from agentlab.policies import ScriptedPolicy
from agentlab.runtime import AgentRuntime
from agentlab.schemas import AgentAction, AgentTask


class RepeatingPolicy:
    name = "repeat"
    model_name = "test"

    def generate(self, state, tools):
        return AgentAction.tool("list_files", {})


class InvalidToolPolicy:
    name = "invalid"
    model_name = "test"

    def generate(self, state, tools):
        if not state.tool_history:
            return AgentAction.tool("shell", {})
        return AgentAction.final("done")


def test_scripted_runtime_success_and_trajectory(tmp_path):
    task = get_task("calculator_fix_divide_zero")
    result = AgentRuntime(
        ScriptedPolicy(), AgentLabConfig(runs_dir=tmp_path / "runs")
    ).run(task)
    assert result.evaluation.success
    assert result.trajectory.total_steps == 6
    assert Path(result.trajectory.trace_path).is_file()
    assert Path(result.trajectory.trace_path).with_name("final_diff.patch").is_file()


def test_repeated_call_and_max_steps(tmp_path):
    task = get_task("calculator_find_tests")
    result = AgentRuntime(
        RepeatingPolicy(), AgentLabConfig(max_steps=3, runs_dir=tmp_path / "runs")
    ).run(task)
    assert result.state.status.value == "max_steps"
    assert any(item["repeated"] for item in result.state.tool_history)
    assert result.evaluation.error_category == "MAX_STEPS"


def test_invalid_tool_is_recorded(tmp_path):
    task = AgentTask(
        "invalid-tool",
        "try invalid",
        get_task("calculator_find_tests").repo_path,
        allowed_tools=["list_files"],
        task_type="query",
        gold={"expected_keywords": ["never"]},
    )
    result = AgentRuntime(
        InvalidToolPolicy(), AgentLabConfig(runs_dir=tmp_path / "runs")
    ).run(task)
    assert result.state.tool_history[0]["error_category"] == "INVALID_TOOL"
    assert not result.evaluation.success
