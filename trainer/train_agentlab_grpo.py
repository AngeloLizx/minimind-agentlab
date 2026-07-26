from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlab.config import AgentLabConfig
from agentlab.evaluation import load_tasks
from agentlab.policies import ScriptedPolicy
from agentlab.runtime import AgentRuntime


def group_relative_advantages(rewards, group_size: int):
    import torch

    if rewards.numel() % group_size:
        raise ValueError("reward count must be divisible by group_size")
    groups = rewards.view(-1, group_size)
    return ((groups - groups.mean(dim=1, keepdim=True)) / (groups.std(dim=1, keepdim=True, unbiased=False) + 1e-6)).view(-1)


def grpo_loss(
    new_logps,
    old_logps,
    advantages,
    ref_logps,
    clip_eps: float = 0.2,
    beta: float = 0.01,
) :
    import torch

    ratio = torch.exp(new_logps - old_logps)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    kl = (torch.exp(ref_logps - new_logps) - (ref_logps - new_logps) - 1).mean()
    return policy + beta * kl


def reward_from_run(result) -> float:
    score = result.evaluation.score
    if result.evaluation.invalid_tool_calls:
        score -= 0.25
    if result.evaluation.repeated_calls:
        score -= 0.25
    if result.evaluation.timed_out:
        score -= 0.5
    return max(-3.0, min(3.0, score))


def smoke_test() -> None:
    tasks = load_tasks("test")[:2]
    rewards = []
    for task in tasks:
        result = AgentRuntime(
            ScriptedPolicy(), AgentLabConfig(max_steps=8, runs_dir=ROOT / "runs")
        ).run(task)
        rewards.extend([reward_from_run(result), reward_from_run(result) - 0.25])
        print(f"rollout task={task.task_id} success={result.evaluation.success}")
    print(f"reward_range=({min(rewards):.3f},{max(rewards):.3f})")
    try:
        import torch
    except ModuleNotFoundError:
        groups = [rewards[index : index + 2] for index in range(0, len(rewards), 2)]
        advantages = []
        for group in groups:
            mean = sum(group) / len(group)
            variance = sum((value - mean) ** 2 for value in group) / len(group)
            scale = variance**0.5 + 1e-6
            advantages.extend((value - mean) / scale for value in group)
        print(f"group_advantages={[round(value, 4) for value in advantages]}")
        print("PyTorch is unavailable: optimizer/KL smoke step and GPU model training were not run.")
        return
    reward_tensor = torch.tensor(rewards)
    advantages = group_relative_advantages(reward_tensor, group_size=2)
    parameter = torch.nn.Parameter(torch.zeros_like(advantages))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    loss = grpo_loss(parameter, torch.zeros_like(parameter), advantages, torch.zeros_like(parameter))
    loss.backward()
    optimizer.step()
    print(f"grpo_objective_step_loss={loss.item():.6f}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print("GPU model training was not run; smoke_test validated rollout, reward, advantage, KL and optimizer flow.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal AgentLab GRPO experiment")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test:
        parser.error("Use --smoke_test; full model training should be launched only after reviewing reward traces.")
    smoke_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
