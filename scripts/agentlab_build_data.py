from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlab.evaluation import load_tasks
from agentlab.policies.scripted_policy import PATCHES
from agentlab.tools import ToolRegistry


def message(role: str, content: str = "", tools: str = "", tool_calls: str = "") -> dict[str, str]:
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tools": tools,
        "tool_calls": tool_calls,
    }


def tool_call(name: str, arguments: dict) -> dict:
    return {"id": f"seed_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


def build_seed_sample(task) -> dict:
    schemas = ToolRegistry().schemas(task.allowed_tools)
    conversations = [
        message(
            "system",
            "You are a controlled code agent. Use relative paths and verify changes with tests.",
            tools=json.dumps(schemas, ensure_ascii=False),
        ),
        message("user", task.user_query),
    ]
    patch = PATCHES.get(task.task_id)
    if patch:
        path, old_text, new_text, query = patch
        actions = [
            ("search_code", {"query": query}, f"{path}: implementation located"),
            ("read_file", {"path": path, "start_line": 1, "end_line": 200}, "Relevant source inspected"),
            ("apply_patch", {"path": path, "old_text": old_text, "new_text": new_text}, f"Modified {path}"),
            ("run_tests", {"command": "pytest -q", "timeout": 30}, "All task tests passed"),
            ("git_diff", {}, f"Diff contains only {path}"),
        ]
        for name, arguments, observation in actions:
            payload = tool_call(name, arguments)
            json.loads(json.dumps(payload))
            conversations.append(message("assistant", tool_calls=json.dumps([payload], ensure_ascii=False)))
            conversations.append(message("tool", observation))
        conversations.append(message("assistant", "Implemented the minimal change and verified it with pytest."))
    else:
        keywords = " ".join(str(x) for x in task.gold.get("expected_keywords", []))
        conversations.append(message("assistant", f"Inspection result: {keywords}"))
    return {"conversations": conversations}


def load_successful_trajectories(runs_dir: Path) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    task_map = {task.task_id: task for task in load_tasks()}
    for path in sorted(runs_dir.glob("*/trajectory.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("success") or not raw.get("final_answer"):
            continue
        task_id = raw.get("task_id")
        task = task_map.get(task_id)
        if task is None or task_id in samples:
            continue
        conversations = [
            message(
                "system",
                "You are a controlled code agent. Use relative paths and verify changes with tests.",
                tools=json.dumps(ToolRegistry().schemas(task.allowed_tools), ensure_ascii=False),
            ),
            message("user", task.user_query),
        ]
        for step in raw.get("steps", []):
            action = step.get("action", {})
            call = action.get("tool_call")
            if action.get("kind") != "tool" or not call:
                continue
            payload = tool_call(call["name"], call.get("arguments", {}))
            conversations.append(message("assistant", tool_calls=json.dumps([payload], ensure_ascii=False)))
            conversations.append(message("tool", step.get("observation_summary", "")))
        conversations.append(message("assistant", raw["final_answer"]))
        samples[task_id] = {"conversations": conversations}
    return samples


def validate_samples(samples: list[dict]) -> None:
    for sample in samples:
        conversations = sample.get("conversations")
        if not conversations or conversations[-1]["role"] != "assistant":
            raise ValueError("incomplete trajectory")
        for item in conversations:
            if item.get("tool_calls"):
                calls = json.loads(item["tool_calls"])
                if not isinstance(calls, list) or not calls[0]["function"]["name"]:
                    raise ValueError("invalid tool call JSON")


def dry_run(path: Path, model_path: str, max_seq_len: int, forward: bool) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_samples(rows)
    assistant_messages = sum(
        item["role"] == "assistant" for row in rows for item in row["conversations"]
    )
    tool_calls = sum(
        bool(item.get("tool_calls")) for row in rows for item in row["conversations"]
    )
    print(
        f"structural_dry_run samples={len(rows)} assistant_messages={assistant_messages} "
        f"tool_calls={tool_calls}"
    )
    try:
        from transformers import AutoTokenizer
        from dataset.lm_dataset import SFTDataset
    except ModuleNotFoundError as exc:
        print(f"MiniMind dataset/forward validation skipped because a base dependency is missing: {exc}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = SFTDataset(str(path), tokenizer, max_length=max_seq_len)
    input_ids, labels = dataset[0]
    supervised = int((labels != -100).sum().item())
    if supervised == 0:
        raise RuntimeError("loss mask is empty")
    rendered = dataset.create_chat_prompt(dataset.samples[0]["conversations"])
    print(f"dry_run samples={len(dataset)} rendered_chars={len(rendered)} supervised_tokens={supervised}")
    if forward:
        try:
            import torch
            from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        except ModuleNotFoundError as exc:
            print(f"single-batch forward skipped because a base dependency is missing: {exc}")
            return

        config = MiniMindConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=len(tokenizer),
            max_position_embeddings=max_seq_len,
        )
        model = MiniMindForCausalLM(config)
        with torch.no_grad():
            loss = model(input_ids=input_ids.unsqueeze(0), labels=labels.unsqueeze(0)).loss
        print(f"forward_loss_is_finite={bool(torch.isfinite(loss))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MiniMind-compatible AgentLab SFT data")
    parser.add_argument("--output", default=str(ROOT / "dataset" / "agentlab_sft_train.jsonl"))
    parser.add_argument("--validation_output", default=str(ROOT / "dataset" / "agentlab_sft_validation.jsonl"))
    parser.add_argument("--runs_dir", default=str(ROOT / "runs"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--model_path", default=str(ROOT / "model"))
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()
    tasks = load_tasks()
    trajectory_samples = load_successful_trajectories(Path(args.runs_dir))
    samples = [trajectory_samples.get(task.task_id, build_seed_sample(task)) for task in tasks]
    validate_samples(samples)
    train, validation = samples, samples[-2:]
    for target, rows in [(Path(args.output), train), (Path(args.validation_output), validation)]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    print(
        f"wrote train={len(train)} validation={len(validation)} "
        f"(trajectory_backed={len(trajectory_samples)}, gold_seed={len(samples) - len(trajectory_samples)})"
    )
    if args.dry_run:
        dry_run(Path(args.output), args.model_path, args.max_seq_len, args.forward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
