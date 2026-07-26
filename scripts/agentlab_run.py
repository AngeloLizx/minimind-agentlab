from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlab.config import AgentLabConfig
from agentlab.evaluation import get_task
from agentlab.policies import MiniMindPolicy, OpenAICompatiblePolicy, ScriptedPolicy
from agentlab.runtime import AgentRuntime
from agentlab.schemas import AgentTask


def build_policy(args):
    if args.policy == "scripted":
        return ScriptedPolicy()
    if args.policy == "minimind":
        if not args.model_path:
            raise SystemExit("--model_path is required for minimind policy")
        return MiniMindPolicy(
            args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            open_thinking=args.open_thinking,
            device=args.device,
        )
    return OpenAICompatiblePolicy(args.base_url, args.api_key, args.model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one MiniMind-AgentLab task")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task_id")
    source.add_argument("--task")
    parser.add_argument("--repo")
    parser.add_argument("--policy", choices=["scripted", "minimind", "openai"], default="scripted")
    parser.add_argument("--model_path")
    parser.add_argument("--model", default="minimind")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="local")
    parser.add_argument("--device")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--open_thinking", action="store_true")
    parser.add_argument("--keep_workspace", action="store_true")
    args = parser.parse_args()
    if args.task_id:
        task = get_task(args.task_id)
    else:
        if not args.repo:
            parser.error("--repo is required with --task")
        task = AgentTask(
            task_id="custom",
            user_query=args.task,
            repo_path=str(Path(args.repo).resolve()),
            allowed_tools=["list_files", "search_code", "read_file", "apply_patch", "run_tests", "git_diff"],
        )
    runtime = AgentRuntime(
        build_policy(args),
        AgentLabConfig(max_steps=args.max_steps, keep_workspace=args.keep_workspace, runs_dir=ROOT / "runs"),
    )
    result = runtime.run(task)
    print("Plan:")
    for step in result.trajectory.initial_plan:
        print(f"  {step['step_id']}. {step['goal']}")
    for step in result.trajectory.steps:
        call = step.action.get("tool_call")
        print(f"Step {step.step}: {step.action.get('kind')}")
        if call:
            print(f"  Tool: {call['name']} {json.dumps(call['arguments'], ensure_ascii=False)}")
            print(f"  Observation: {step.observation_summary[:300]}")
            print(f"  Reflection: {step.reflection}")
    print(f"Final result: {result.state.final_answer or result.state.error}")
    print(f"Test result: {result.trajectory.test_result}")
    print(f"Success: {result.evaluation.success}, score={result.evaluation.score:.3f}")
    print(f"Trace path: {result.trajectory.trace_path}")
    return 0 if result.evaluation.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
