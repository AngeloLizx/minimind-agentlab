from __future__ import annotations

from agentlab.evaluation.error_taxonomy import ErrorCategory
from agentlab.schemas import AgentState, AgentTask, AgentTrajectory, EvaluationResult


class Evaluator:
    def evaluate(self, task: AgentTask, state: AgentState, trajectory: AgentTrajectory) -> EvaluationResult:
        history = state.tool_history
        valid = sum(1 for item in history if item.get("valid", False))
        invalid = len(history) - valid
        repeated = sum(1 for item in history if item.get("repeated", False))
        test_entries = [item for item in history if item.get("name") == "run_tests"]
        test_passed: bool | None = None
        if test_entries:
            test_passed = bool(test_entries[-1].get("success"))

        expected = set(task.gold.get("modified_files", []))
        changed = set(trajectory.modified_files)
        unauthorized = changed - expected if expected else set()
        patch_valid = bool(changed & expected) and not unauthorized if expected else not unauthorized

        if task.task_type in {"explanation", "query", "code_location"}:
            keywords = [str(x).lower() for x in task.gold.get("expected_keywords", [])]
            success = bool(state.final_answer) and all(word in state.final_answer.lower() for word in keywords)
            patch_valid = not changed
        else:
            success = test_passed is True and patch_valid and invalid == 0

        category = ""
        if not success:
            if state.status.value == "timeout":
                category = ErrorCategory.TIMEOUT.value
            elif state.status.value == "max_steps":
                category = ErrorCategory.MAX_STEPS.value
            elif any(item.get("error_category") == "PATH_VIOLATION" for item in history):
                category = ErrorCategory.PATH_VIOLATION.value
            elif any(item.get("error_category") == "INVALID_ARGUMENT" for item in history):
                category = ErrorCategory.INVALID_ARGUMENT.value
            elif any(item.get("error_category") == "INVALID_TOOL" for item in history):
                category = ErrorCategory.INVALID_TOOL.value
            elif repeated:
                category = ErrorCategory.REPEATED_ACTION.value
            elif test_passed is False:
                category = ErrorCategory.TEST_FAILURE.value
            elif state.final_answer and task.task_type not in {"explanation", "query", "code_location"}:
                category = ErrorCategory.HALLUCINATED_SUCCESS.value
            else:
                category = ErrorCategory.UNKNOWN.value
        score = (
            (1.5 if success else 0.0)
            + (0.75 if test_passed is True else 0.0)
            + (0.5 if patch_valid else 0.0)
            + (0.25 if invalid == 0 else -0.25)
            - min(0.5, repeated * 0.25)
        )
        score = max(-3.0, min(3.0, score))
        return EvaluationResult(
            task_id=task.task_id,
            success=success,
            score=score,
            test_passed=test_passed,
            patch_valid=patch_valid,
            valid_tool_calls=valid,
            invalid_tool_calls=invalid,
            repeated_calls=repeated,
            total_steps=trajectory.total_steps,
            latency_seconds=trajectory.total_latency,
            timed_out=state.status.value == "timeout",
            error_category=category,
            details={"changed_files": sorted(changed), "unauthorized_files": sorted(unauthorized)},
        )
