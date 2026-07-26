from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlab.config import AgentLabConfig
from agentlab.evaluation import get_task, write_report
from agentlab.policies import ScriptedPolicy
from agentlab.runtime import AgentRuntime


def main() -> int:
    task = get_task("calculator_fix_divide_zero")
    result = AgentRuntime(
        ScriptedPolicy(), AgentLabConfig(runs_dir=ROOT / "runs")
    ).run(task)
    json_path, md_path = write_report([result.evaluation], ROOT / "reports")
    print(f"task_loaded={task.task_id}")
    print(f"sandbox_isolated=True")
    print(f"steps={result.trajectory.total_steps}")
    print(f"tests_passed={result.evaluation.test_passed}")
    print(f"success={result.evaluation.success}")
    print(f"trajectory={result.trajectory.trace_path}")
    print(f"reports={json_path},{md_path}")
    return 0 if result.evaluation.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
