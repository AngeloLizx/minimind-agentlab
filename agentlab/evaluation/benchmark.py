from __future__ import annotations

import json
from pathlib import Path

from agentlab.schemas import AgentTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "agentlab"


def load_tasks(split: str | None = None) -> list[AgentTask]:
    splits = [split] if split else ["train", "dev", "test"]
    tasks: list[AgentTask] = []
    for name in splits:
        path = BENCHMARK_ROOT / "tasks" / f"{name}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                validator = dict(raw.get("validator", {}))
                if validator.get("test_file"):
                    validator["test_file"] = str((BENCHMARK_ROOT / validator["test_file"]).resolve())
                tasks.append(
                    AgentTask(
                        task_id=raw["task_id"],
                        user_query=raw["prompt"],
                        repo_path=str((BENCHMARK_ROOT / "repos" / raw["repo_id"]).resolve()),
                        repo_id=raw["repo_id"],
                        task_type=raw.get("task_type", "bug_fix"),
                        difficulty=raw.get("difficulty", 1),
                        allowed_tools=raw.get("allowed_tools", []),
                        validator=validator,
                        gold=raw.get("gold", {}),
                        split=raw.get("split", name),
                    )
                )
    return tasks


def get_task(task_id: str) -> AgentTask:
    for task in load_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task: {task_id}")
