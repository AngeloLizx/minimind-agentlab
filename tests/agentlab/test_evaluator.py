from __future__ import annotations

import json

from agentlab.evaluation.metrics import aggregate_metrics
from agentlab.evaluation.report import write_report
from agentlab.schemas import EvaluationResult


def result(task_id: str, success: bool, category: str = "") -> EvaluationResult:
    return EvaluationResult(
        task_id=task_id,
        success=success,
        score=3.0 if success else 0.0,
        test_passed=success,
        patch_valid=success,
        valid_tool_calls=5,
        invalid_tool_calls=0 if success else 1,
        repeated_calls=0,
        total_steps=6,
        latency_seconds=0.5,
        timed_out=False,
        error_category=category,
    )


def test_metrics_and_reports_are_derived_from_results(tmp_path):
    rows = [result("ok", True), result("bad", False, "TEST_FAILURE")]
    metrics = aggregate_metrics(rows)
    assert metrics["task_success_rate"] == 0.5
    assert metrics["test_pass_rate"] == 0.5
    assert metrics["error_distribution"] == {"TEST_FAILURE": 1}
    json_path, md_path = write_report(rows, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["tasks"]) == 2
    assert "TEST_FAILURE" in md_path.read_text(encoding="utf-8")
