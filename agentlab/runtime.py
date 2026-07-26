from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .config import AgentLabConfig
from .context_manager import ContextManager
from .env.sandbox import Sandbox
from .evaluation.evaluator import Evaluator
from .planner import SimplePlanner
from .policies.base import AgentPolicy
from .reflection import RuleBasedReflection
from .schemas import (
    AgentRunResult,
    AgentState,
    AgentStatus,
    AgentTask,
    AgentTrajectory,
    ToolResult,
    TrajectoryStep,
    utc_now,
)
from .tools.registry import ToolRegistry
from .trajectory import TrajectoryStore


class AgentRuntime:
    def __init__(
        self,
        policy: AgentPolicy,
        config: AgentLabConfig | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.policy = policy
        self.config = config or AgentLabConfig()
        self.registry = registry or ToolRegistry()
        self.planner = SimplePlanner()
        self.reflection = RuleBasedReflection()
        self.context = ContextManager(self.config)
        self.evaluator = Evaluator()
        self.store = TrajectoryStore(self.config.runs_dir)

    def run(self, task: AgentTask) -> AgentRunResult:
        run_id = f"{task.task_id}-{uuid.uuid4().hex[:10]}"
        state = AgentState(
            task_id=task.task_id,
            user_query=task.user_query,
            repo_path=task.repo_path,
            max_steps=self.config.max_steps,
            status=AgentStatus.RUNNING,
        )
        state.plan = self.planner.create_plan(task)
        trajectory = AgentTrajectory(
            run_id=run_id,
            task_id=task.task_id,
            policy=self.policy.name,
            model=getattr(self.policy, "model_name", ""),
            initial_plan=[asdict(step) for step in state.plan],
        )
        started = time.perf_counter()
        last_signatures: set[str] = set()
        no_progress = 0
        final_diff = ""

        with Sandbox(task.repo_path, keep=self.config.keep_workspace) as sandbox:
            sandbox.install_validator(task.validator.get("test_file"))
            state.repo_path = str(sandbox.root)
            try:
                while state.step_count < self.config.max_steps:
                    if time.perf_counter() - started > self.config.task_timeout_seconds:
                        state.status = AgentStatus.TIMEOUT
                        state.error = "task timeout"
                        break
                    state.messages = self.context.build_messages(state)
                    action = self.policy.generate(state, self.registry.schemas(task.allowed_tools))
                    state.step_count += 1
                    action_dict = asdict(action)

                    if action.kind == "final":
                        state.final_answer = action.final_answer.strip()
                        state.status = AgentStatus.SUCCEEDED
                        for step in state.plan:
                            if step.status == "pending":
                                step.status = "completed"
                                step.notes = "Policy returned the final answer."
                                state.completed_steps.append(step.step_id)
                        trajectory.steps.append(TrajectoryStep(state.step_count, action_dict))
                        break
                    if action.kind != "tool" or action.tool_call is None:
                        result = ToolResult(
                            "",
                            False,
                            error=action.final_answer or "policy returned an invalid action",
                            error_category="MODEL_ERROR",
                        )
                    else:
                        call = action.tool_call
                        signature = json.dumps([call.name, call.arguments], sort_keys=True, ensure_ascii=False)
                        repeated = signature in last_signatures
                        allowed = not task.allowed_tools or call.name in task.allowed_tools
                        if repeated:
                            result = ToolResult(
                                call.name,
                                False,
                                error="identical tool call already executed",
                                error_category="REPEATED_ACTION",
                            )
                        elif not allowed:
                            result = ToolResult(
                                call.name,
                                False,
                                error=f"tool not allowed for task: {call.name}",
                                error_category="INVALID_TOOL",
                            )
                        else:
                            if call.name == "run_tests":
                                call.arguments["timeout"] = min(
                                    int(call.arguments.get("timeout", self.config.tool_timeout_seconds)),
                                    int(self.config.tool_timeout_seconds),
                                )
                            result = self.registry.execute(call.name, sandbox, call.arguments)
                            if action_dict.get("tool_call"):
                                action_dict["tool_call"]["arguments"] = dict(call.arguments)
                        last_signatures.add(signature)

                    output_ref = self.store.save_tool_output(run_id, state.step_count, result)
                    result.output_ref = output_ref
                    truncated = result.output[: self.config.max_tool_result_chars]
                    history_item = {
                        "name": result.tool_name,
                        "arguments": action.tool_call.arguments if action.tool_call else {},
                        "success": result.success,
                        "valid": result.success or result.error_category in {"TEST_FAILURE"},
                        "repeated": result.error_category == "REPEATED_ACTION",
                        "error_category": result.error_category,
                        "summary": result.summary,
                        "output_ref": output_ref,
                        "metadata": result.metadata,
                    }
                    state.tool_history.append(history_item)
                    state.observations.append(truncated or result.error)
                    pending = next((step for step in state.plan if step.status == "pending"), None)
                    if pending is not None:
                        if result.success:
                            pending.status = "completed"
                            pending.notes = result.summary[:200]
                            state.completed_steps.append(pending.step_id)
                        else:
                            pending.notes = result.error[:200]
                    repeated_flag = result.error_category == "REPEATED_ACTION"
                    reflection = self.reflection.reflect(state, result, repeated_flag)
                    trajectory.steps.append(
                        TrajectoryStep(
                            step=state.step_count,
                            action=action_dict,
                            tool_result={
                                "success": result.success,
                                "summary": result.summary,
                                "latency_seconds": result.latency_seconds,
                                "error_category": result.error_category,
                                "output_ref": output_ref,
                            },
                            observation_summary=result.summary,
                            reflection=reflection,
                        )
                    )
                    no_progress = no_progress + 1 if repeated_flag or not (result.output or "").strip() else 0
                    if no_progress >= self.config.no_progress_limit:
                        state.working_memory["no_progress"] = True
                    if result.tool_name == "run_tests":
                        trajectory.test_result = {**result.metadata, "output": result.output}
                    if result.tool_name == "git_diff" and result.success:
                        final_diff = result.output
                else:
                    state.status = AgentStatus.MAX_STEPS
                    state.error = "maximum steps reached"
            except Exception as exc:
                state.status = AgentStatus.FAILED
                state.error = f"{type(exc).__name__}: {exc}"

            if not final_diff:
                final_diff = sandbox.diff()
            trajectory.modified_files = [
                name for name in sandbox.changed_files() if name != "tests/test_agentlab_task.py"
            ]
            state.end_time = utc_now()
            trajectory.final_answer = state.final_answer
            trajectory.total_steps = state.step_count
            trajectory.total_latency = time.perf_counter() - started
            evaluation = self.evaluator.evaluate(task, state, trajectory)
            trajectory.success = evaluation.success
            trajectory.error_category = evaluation.error_category
            if evaluation.success:
                state.status = AgentStatus.SUCCEEDED
            elif state.status == AgentStatus.SUCCEEDED:
                state.status = AgentStatus.FAILED
            self.store.save(trajectory, state, evaluation, final_diff)
        return AgentRunResult(state, trajectory, evaluation)
