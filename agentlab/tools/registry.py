from __future__ import annotations

from typing import Iterable

from agentlab.env.sandbox import Sandbox
from agentlab.schemas import ToolResult

from .apply_patch import ApplyPatchTool
from .base import AgentTool
from .git_diff import GitDiffTool
from .list_files import ListFilesTool
from .read_file import ReadFileTool
from .run_tests import RunTestsTool
from .search_code import SearchCodeTool


class ToolRegistry:
    def __init__(self, tools: Iterable[AgentTool] | None = None):
        tools = tools or [
            ListFilesTool(),
            SearchCodeTool(),
            ReadFileTool(),
            ApplyPatchTool(),
            RunTestsTool(),
            GitDiffTool(),
        ]
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self, allowed: list[str] | None = None) -> list[dict]:
        names = set(allowed or self._tools)
        return [tool.schema() for name, tool in self._tools.items() if name in names]

    def execute(self, name: str, sandbox: Sandbox, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, error=f"unknown tool: {name}", error_category="INVALID_TOOL")
        return tool.execute(sandbox, arguments)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
