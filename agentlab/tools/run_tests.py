from __future__ import annotations

import re
import subprocess
import sys
from time import perf_counter

from agentlab.env.sandbox import Sandbox
from agentlab.schemas import ToolResult

from .base import AgentTool


class RunTestsTool(AgentTool):
    name = "run_tests"
    description = "Run one allow-listed test command without a shell."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["pytest -q", "python -m pytest -q", "python -m compileall ."],
            },
            "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def run(self, sandbox: Sandbox, command: str, timeout: int = 30) -> ToolResult:
        assert sandbox.root
        argv = {
            "pytest -q": [sys.executable, "-m", "pytest", "-q"],
            "python -m pytest -q": [sys.executable, "-m", "pytest", "-q"],
            "python -m compileall .": [sys.executable, "-m", "compileall", "."],
        }[command]
        started = perf_counter()
        try:
            proc = subprocess.run(
                argv,
                cwd=sandbox.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[:12_000]
            return ToolResult(
                self.name,
                False,
                output=output,
                error=f"test command timed out after {timeout}s",
                metadata={"timeout": True, "returncode": None, "duration": perf_counter() - started},
                error_category="TIMEOUT",
            )
        output = (proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))[:12_000]
        passed = sum(int(x) for x in re.findall(r"(\d+) passed", output))
        failed = sum(int(x) for x in re.findall(r"(\d+) failed", output))
        return ToolResult(
            self.name,
            proc.returncode == 0,
            output=output,
            error="" if proc.returncode == 0 else "tests failed",
            metadata={
                "returncode": proc.returncode,
                "passed": passed,
                "failed": failed,
                "duration": perf_counter() - started,
                "timeout": False,
            },
            error_category="" if proc.returncode == 0 else "TEST_FAILURE",
        )
