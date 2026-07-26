from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AgentLabConfig:
    """Runtime limits. Conservative defaults keep local experiments bounded."""

    max_steps: int = 12
    task_timeout_seconds: float = 120.0
    tool_timeout_seconds: float = 30.0
    max_context_chars: int = 24_000
    max_tool_result_chars: int = 8_000
    recent_steps: int = 4
    max_repeated_calls: int = 1
    no_progress_limit: int = 2
    keep_workspace: bool = False
    runs_dir: Path = Path("runs")

