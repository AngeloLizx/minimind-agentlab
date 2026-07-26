from __future__ import annotations

import json
from dataclasses import asdict

from .config import AgentLabConfig
from .schemas import AgentState


class ContextManager:
    def __init__(self, config: AgentLabConfig):
        self.config = config

    def build_messages(self, state: AgentState) -> list[dict]:
        plan = [asdict(step) for step in state.plan]
        recent = state.tool_history[-self.config.recent_steps :]
        content = {
            "task": state.user_query,
            "plan": plan,
            "working_memory": state.working_memory,
            "recent_tool_history": recent,
        }
        context = json.dumps(content, ensure_ascii=False)
        if len(context) > self.config.max_context_chars:
            context = context[: self.config.max_context_chars] + "\n[context truncated]"
        system = {
            "role": "system",
            "content": (
                "You are a controlled code agent. Use only provided tools, relative paths, "
                "minimal edits, and test evidence. Return a final answer when done."
            ),
        }
        return [system, {"role": "user", "content": context}]
