from __future__ import annotations

from agentlab.env.sandbox import Sandbox
from agentlab.schemas import ToolResult

from .base import AgentTool


class GitDiffTool(AgentTool):
    name = "git_diff"
    description = "Show changes relative to the sandbox's initial copied state."
    parameters = {
        "type": "object",
        "properties": {"max_chars": {"type": "integer", "minimum": 100, "maximum": 50_000}},
        "additionalProperties": False,
    }

    def run(self, sandbox: Sandbox, max_chars: int = 20_000) -> ToolResult:
        changed = [name for name in sandbox.changed_files() if not name.endswith("test_agentlab_task.py")]
        return ToolResult(
            self.name,
            True,
            output=sandbox.diff(max_chars=max_chars),
            metadata={"modified_files": changed},
        )
