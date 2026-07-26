from __future__ import annotations

import json
import re
from typing import Any

from agentlab.schemas import AgentAction, AgentState


class MiniMindPolicy:
    name = "minimind"

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        open_thinking: bool = False,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.open_thinking = open_thinking
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().to(self.device)

    @staticmethod
    def parse_response(text: str) -> AgentAction:
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if matches:
            try:
                payload = json.loads(matches[0])
                arguments = payload.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                return AgentAction.tool(str(payload.get("name", "")), arguments)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                return AgentAction(kind="invalid", raw_text=text, final_answer=f"Invalid tool call: {exc}")
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return AgentAction.final(answer)

    def generate(self, state: AgentState, tools: list[dict]) -> AgentAction:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            state.messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            open_thinking=self.open_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": max(self.temperature, 1e-5),
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            output = self.model.generate(**kwargs)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        action = self.parse_response(text)
        action.raw_text = text
        return action
