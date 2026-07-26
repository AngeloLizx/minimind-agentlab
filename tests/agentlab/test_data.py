from __future__ import annotations

import json
from pathlib import Path


def test_generated_sft_data_is_complete_and_tool_json_is_valid():
    path = Path(__file__).resolve().parents[2] / "dataset" / "agentlab_sft_train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert 20 <= len(rows) <= 50
    for row in rows:
        conversations = row["conversations"]
        assert conversations[0]["role"] == "system"
        assert conversations[-1]["role"] == "assistant"
        for item in conversations:
            assert {"role", "content", "reasoning_content", "tools", "tool_calls"} <= set(item)
            if item["tool_calls"]:
                calls = json.loads(item["tool_calls"])
                assert calls[0]["function"]["name"]
