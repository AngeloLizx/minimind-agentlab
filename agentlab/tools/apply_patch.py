from __future__ import annotations

import difflib

from agentlab.env.sandbox import Sandbox
from agentlab.schemas import ToolResult

from .base import AgentTool


class ApplyPatchTool(AgentTool):
    name = "apply_patch"
    description = "Replace one unique UTF-8 text fragment in a sandbox file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def run(self, sandbox: Sandbox, path: str, old_text: str, new_text: str) -> ToolResult:
        file = sandbox.resolve(path)
        if not file.is_file() or file.is_symlink():
            return ToolResult(self.name, False, error="path must be a regular file", error_category="PATH_VIOLATION")
        try:
            before = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(self.name, False, error="binary files cannot be patched", error_category="PATCH_ERROR")
        count = before.count(old_text)
        if not old_text or count != 1:
            return ToolResult(
                self.name,
                False,
                error=f"old_text must occur exactly once (found {count})",
                error_category="PATCH_ERROR",
            )
        after = before.replace(old_text, new_text, 1)
        backup = file.with_suffix(file.suffix + ".agentlab.bak")
        backup.write_bytes(before.encode("utf-8"))
        file.write_bytes(after.encode("utf-8"))
        backup.unlink()
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                f"a/{path}",
                f"b/{path}",
            )
        )
        return ToolResult(self.name, True, output=diff[:8_000], metadata={"modified_files": [path]})
