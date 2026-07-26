from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from agentlab.env.sandbox import PathViolation, Sandbox
from agentlab.env.validators import validate_json_arguments
from agentlab.schemas import ToolResult


class AgentTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, sandbox: Sandbox, arguments: dict[str, Any]) -> ToolResult:
        started = perf_counter()
        errors = validate_json_arguments(arguments, self.parameters)
        if errors:
            return ToolResult(
                self.name,
                False,
                error="; ".join(errors),
                latency_seconds=perf_counter() - started,
                error_category="INVALID_ARGUMENT",
            )
        try:
            result = self.run(sandbox, **arguments)
            result.latency_seconds = perf_counter() - started
            return result
        except PathViolation as exc:
            return ToolResult(
                self.name,
                False,
                error=str(exc),
                latency_seconds=perf_counter() - started,
                error_category="PATH_VIOLATION",
            )
        except Exception as exc:
            return ToolResult(
                self.name,
                False,
                error=f"{type(exc).__name__}: {exc}",
                latency_seconds=perf_counter() - started,
                error_category="UNKNOWN",
            )

    @abstractmethod
    def run(self, sandbox: Sandbox, **arguments: Any) -> ToolResult:
        raise NotImplementedError
