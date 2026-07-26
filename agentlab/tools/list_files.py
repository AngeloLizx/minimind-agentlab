from __future__ import annotations

from pathlib import Path

from agentlab.env.sandbox import Sandbox
from agentlab.schemas import ToolResult

from .base import AgentTool


class ListFilesTool(AgentTool):
    name = "list_files"
    description = "List repository files below a relative directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_depth": {"type": "integer", "minimum": 0, "maximum": 10},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        "additionalProperties": False,
    }
    ignored = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}

    def run(self, sandbox: Sandbox, path: str = ".", max_depth: int = 3, max_results: int = 200) -> ToolResult:
        base = sandbox.resolve(path)
        if not base.is_dir():
            return ToolResult(self.name, False, error="path is not a directory", error_category="INVALID_ARGUMENT")
        rows: list[str] = []
        root = sandbox.root
        assert root
        for item in sorted(base.rglob("*")):
            rel_parts = item.relative_to(base).parts
            if any(part in self.ignored for part in rel_parts) or len(rel_parts) > max_depth:
                continue
            if item.is_symlink():
                try:
                    sandbox.resolve(item.relative_to(root))
                except Exception:
                    continue
            suffix = "/" if item.is_dir() else ""
            rows.append(item.relative_to(root).as_posix() + suffix)
            if len(rows) >= max_results:
                break
        return ToolResult(self.name, True, output="\n".join(rows), metadata={"count": len(rows)})
