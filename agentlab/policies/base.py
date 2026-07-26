from __future__ import annotations

from typing import Protocol

from agentlab.schemas import AgentAction, AgentState


class AgentPolicy(Protocol):
    name: str
    model_name: str

    def generate(self, state: AgentState, tools: list[dict]) -> AgentAction:
        ...
