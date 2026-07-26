from __future__ import annotations

import re

from agentlab.env.sandbox import Sandbox
from agentlab.schemas import ToolResult

from .base import AgentTool


class SearchCodeTool(AgentTool):
    name = "search_code"
    description = "Search UTF-8 text files for a keyword or regular expression."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "path": {"type": "string"},
            "regex": {"type": "boolean"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    extensions = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".ini", ".cfg"}

    def run(
        self,
        sandbox: Sandbox,
        query: str,
        path: str = ".",
        regex: bool = False,
        max_results: int = 50,
    ) -> ToolResult:
        if not query:
            return ToolResult(self.name, False, error="query cannot be empty", error_category="INVALID_ARGUMENT")
        base = sandbox.resolve(path)
        try:
            pattern = re.compile(query if regex else re.escape(query))
        except re.error as exc:
            return ToolResult(self.name, False, error=f"invalid regex: {exc}", error_category="INVALID_ARGUMENT")
        files = [base] if base.is_file() else base.rglob("*")
        matches: list[str] = []
        assert sandbox.root
        for file in files:
            if not file.is_file() or file.is_symlink() or file.suffix.lower() not in self.extensions:
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(lines, 1):
                if pattern.search(line):
                    matches.append(f"{file.relative_to(sandbox.root).as_posix()}:{number}: {line.strip()[:300]}")
                    if len(matches) >= max_results:
                        return ToolResult(self.name, True, "\n".join(matches), metadata={"count": len(matches)})
        return ToolResult(self.name, True, "\n".join(matches), metadata={"count": len(matches)})
