from __future__ import annotations

import json

from agentlab.policies.minimind_policy import MiniMindPolicy
from agentlab.schemas import AgentAction, AgentState


class OpenAICompatiblePolicy:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2):
        from openai import OpenAI

        self.model_name = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, state: AgentState, tools: list[dict]) -> AgentAction:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=state.messages,
            tools=tools,
            temperature=self.temperature,
        )
        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            arguments = json.loads(call.function.arguments or "{}")
            return AgentAction.tool(call.function.name, arguments)
        return MiniMindPolicy.parse_response(message.content or "")
