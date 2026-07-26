from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_STEPS = "max_steps"


@dataclass(slots=True)
class PlanStep:
    step_id: int
    goal: str
    status: str = "pending"
    notes: str = ""


@dataclass(slots=True)
class AgentTask:
    task_id: str
    user_query: str
    repo_path: str
    repo_id: str = ""
    task_type: str = "bug_fix"
    difficulty: int = 1
    allowed_tools: list[str] = field(default_factory=list)
    validator: dict[str, Any] = field(default_factory=dict)
    gold: dict[str, Any] = field(default_factory=dict)
    split: str = "custom"


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentAction:
    kind: str
    tool_call: ToolCall | None = None
    final_answer: str = ""
    raw_text: str = ""

    @classmethod
    def tool(cls, name: str, arguments: dict[str, Any]) -> "AgentAction":
        return cls(kind="tool", tool_call=ToolCall(name, arguments))

    @classmethod
    def final(cls, answer: str) -> "AgentAction":
        return cls(kind="final", final_answer=answer)


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error_category: str = ""
    output_ref: str = ""

    @property
    def summary(self) -> str:
        text = self.output
        if self.error:
            text = f"{self.error}\n{text}" if text else self.error
        return text[:800]


@dataclass(slots=True)
class TrajectoryStep:
    step: int
    action: dict[str, Any]
    tool_result: dict[str, Any] | None = None
    observation_summary: str = ""
    reflection: str = ""
    timestamp: str = field(default_factory=utc_now)


@dataclass(slots=True)
class AgentState:
    task_id: str
    user_query: str
    repo_path: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    plan: list[PlanStep] = field(default_factory=list)
    completed_steps: list[int] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 12
    status: AgentStatus = AgentStatus.PENDING
    final_answer: str = ""
    error: str = ""
    start_time: str = field(default_factory=utc_now)
    end_time: str = ""


@dataclass(slots=True)
class AgentTrajectory:
    run_id: str
    task_id: str
    policy: str
    model: str = ""
    success: bool = False
    initial_plan: list[dict[str, Any]] = field(default_factory=list)
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    modified_files: list[str] = field(default_factory=list)
    test_result: dict[str, Any] = field(default_factory=dict)
    total_steps: int = 0
    total_latency: float = 0.0
    error_category: str = ""
    trace_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationResult:
    task_id: str
    success: bool
    score: float
    test_passed: bool | None
    patch_valid: bool
    valid_tool_calls: int
    invalid_tool_calls: int
    repeated_calls: int
    total_steps: int
    latency_seconds: float
    timed_out: bool
    error_category: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRunResult:
    state: AgentState
    trajectory: AgentTrajectory
    evaluation: EvaluationResult
