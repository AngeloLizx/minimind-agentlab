from __future__ import annotations

from collections import Counter
from statistics import mean

from agentlab.schemas import EvaluationResult


def aggregate_metrics(results: list[EvaluationResult]) -> dict:
    if not results:
        return {}
    calls = sum(x.valid_tool_calls + x.invalid_tool_calls for x in results)
    valid = sum(x.valid_tool_calls for x in results)
    repeated = sum(x.repeated_calls for x in results)
    tests = [x for x in results if x.test_passed is not None]
    return {
        "task_success_rate": mean(x.success for x in results),
        "test_pass_rate": mean(x.test_passed is True for x in tests) if tests else None,
        "valid_tool_call_rate": valid / calls if calls else 1.0,
        "invalid_tool_call_rate": (calls - valid) / calls if calls else 0.0,
        "argument_validity_rate": valid / calls if calls else 1.0,
        "repeated_call_rate": repeated / calls if calls else 0.0,
        "average_steps": mean(x.total_steps for x in results),
        "average_latency": mean(x.latency_seconds for x in results),
        "timeout_rate": mean(x.timed_out for x in results),
        "patch_validity_rate": mean(x.patch_valid for x in results),
        "error_distribution": dict(Counter(x.error_category for x in results if x.error_category)),
    }
