from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schemas import AgentState, AgentTrajectory, EvaluationResult, ToolResult


class TrajectoryStore:
    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir)

    def prepare(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        (path / "tool_outputs").mkdir(parents=True, exist_ok=True)
        return path

    def save_tool_output(self, run_id: str, step: int, result: ToolResult) -> str:
        path = self.prepare(run_id) / "tool_outputs" / f"step_{step:03d}_{result.tool_name}.txt"
        path.write_text(result.output or result.error, encoding="utf-8")
        return str(path)

    def save(
        self,
        trajectory: AgentTrajectory,
        state: AgentState,
        evaluation: EvaluationResult,
        diff: str,
    ) -> Path:
        run_dir = self.prepare(trajectory.run_id)
        trajectory.trace_path = str(run_dir / "trajectory.json")
        (run_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "final_diff.patch").write_text(diff, encoding="utf-8")
        (run_dir / "test_output.txt").write_text(
            str(trajectory.test_result.get("output", "")), encoding="utf-8"
        )
        summary = {
            "run_id": trajectory.run_id,
            "task_id": trajectory.task_id,
            "state": asdict(state),
            "evaluation": evaluation.to_dict(),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return run_dir
