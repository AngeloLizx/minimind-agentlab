from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agentlab.evaluation.metrics import aggregate_metrics
from agentlab.schemas import EvaluationResult


def write_report(results: list[EvaluationResult], output_dir: str | Path = "reports") -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output / f"agentlab_eval_{stamp}.json"
    md_path = output / f"agentlab_eval_{stamp}.md"
    metrics = aggregate_metrics(results)
    payload = {"metrics": metrics, "tasks": [item.to_dict() for item in results]}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# MiniMind-AgentLab Evaluation", "", "## Metrics", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in metrics.items())
    lines.extend(["", "## Tasks", "", "| task | success | score | error |", "|---|---:|---:|---|"])
    lines.extend(
        f"| {item.task_id} | {item.success} | {item.score:.3f} | {item.error_category or '-'} |"
        for item in results
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
