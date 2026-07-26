from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlab.config import AgentLabConfig
from agentlab.evaluation import load_tasks, write_report
from agentlab.policies import MiniMindPolicy, ScriptedPolicy
from agentlab.runtime import AgentRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MiniMind-AgentLab")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--policy", choices=["scripted", "minimind"], default="scripted")
    parser.add_argument("--model_path")
    parser.add_argument("--max_steps", type=int, default=12)
    args = parser.parse_args()
    if args.policy == "minimind" and not args.model_path:
        parser.error("--model_path is required for minimind policy")
    policy = ScriptedPolicy() if args.policy == "scripted" else MiniMindPolicy(args.model_path)
    evaluations = []
    for task in load_tasks(args.split):
        result = AgentRuntime(
            policy, AgentLabConfig(max_steps=args.max_steps, runs_dir=ROOT / "runs")
        ).run(task)
        evaluations.append(result.evaluation)
        print(f"{task.task_id}: success={result.evaluation.success} score={result.evaluation.score:.3f}")
    json_path, md_path = write_report(evaluations, ROOT / "reports")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0 if all(item.success for item in evaluations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
