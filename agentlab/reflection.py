from __future__ import annotations

from .schemas import AgentState, ToolResult


class RuleBasedReflection:
    """Produces concise operational memory, never hidden chain-of-thought."""

    def reflect(self, state: AgentState, result: ToolResult, repeated: bool = False) -> str:
        reason = ""
        next_step = "Continue with the next plan step."
        avoid = ""
        if repeated:
            reason, next_step, avoid = (
                "The same tool call was repeated.",
                "Use a different query or act on existing evidence.",
                "Do not repeat identical arguments.",
            )
        elif not result.success:
            reason = result.error or "Tool execution failed."
            next_step = "Correct the arguments or choose a safer tool."
        elif result.tool_name == "run_tests" and result.metadata.get("returncode", 0) != 0:
            reason = "Tests failed."
            next_step = "Inspect the failure and make a minimal corrective patch."
        elif not result.output.strip():
            reason = "The tool returned no new information."
            next_step = "Broaden the search or inspect a known file."
        reflection = (
            f"当前目标: {state.plan[min(state.step_count, len(state.plan) - 1)].goal if state.plan else '完成任务'}; "
            f"失败原因: {reason or '无'}; 下一步建议: {next_step}; 避免重复: {avoid or '无'}"
        )
        state.working_memory.update(
            {
                "current_goal": state.plan[min(state.step_count, len(state.plan) - 1)].goal if state.plan else "",
                "failure_reason": reason,
                "next_step": next_step,
                "avoid": avoid,
            }
        )
        return reflection

