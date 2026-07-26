from __future__ import annotations

from agentlab.env.sandbox import Sandbox
from agentlab.schemas import ToolResult

from .base import AgentTool


class ReadFileTool(AgentTool):
    name = "read_file"
    description = "Read a UTF-8 file with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 50_000},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def run(
        self,
        sandbox: Sandbox,
        path: str,
        start_line: int = 1,
        end_line: int = 10_000,
        max_chars: int = 12_000,
    ) -> ToolResult:
        file = sandbox.resolve(path)
        if not file.is_file():
            return ToolResult(self.name, False, error="path is not a file", error_category="INVALID_ARGUMENT")
        if end_line < start_line:
            return ToolResult(self.name, False, error="end_line must be >= start_line", error_category="INVALID_ARGUMENT")
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return ToolResult(self.name, False, error="binary or non-UTF-8 file", error_category="INVALID_ARGUMENT")
        selected = [f"{i:4d} | {lines[i - 1]}" for i in range(start_line, min(end_line, len(lines)) + 1)]
        output = "\n".join(selected)
        truncated = len(output) > max_chars
        output = output[:max_chars] + ("\n[output truncated]" if truncated else "")
        return ToolResult(
            self.name,
            True,
            output=output,
            metadata={"start_line": start_line, "end_line": min(end_line, len(lines)), "truncated": truncated},
        )
