from __future__ import annotations

from .schemas import AgentTask, PlanStep


class SimplePlanner:
    """A deterministic plan keeps small-model behaviour easy to evaluate."""

    def create_plan(self, task: AgentTask) -> list[PlanStep]:
        if task.task_type in {"explanation", "code_location", "query"}:
            goals = ["Inspect repository", "Locate relevant code", "Read the implementation", "Answer with evidence"]
        else:
            goals = [
                "Inspect repository",
                "Locate relevant code",
                "Inspect related tests",
                "Apply minimal patch",
                "Run tests",
                "Review diff",
            ]
        return [PlanStep(i + 1, goal) for i, goal in enumerate(goals)]

