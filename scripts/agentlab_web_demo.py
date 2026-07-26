from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlab.config import AgentLabConfig
from agentlab.evaluation import get_task, load_tasks
from agentlab.policies import MiniMindPolicy, ScriptedPolicy
from agentlab.runtime import AgentRuntime


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="MiniMind-AgentLab", layout="wide")
    st.title("MiniMind-AgentLab")
    tasks = load_tasks()
    task_ids = [task.task_id for task in tasks]
    selected = st.selectbox("任务", task_ids)
    task = get_task(selected)
    st.text_input("仓库", task.repo_path, disabled=True)
    policy_name = st.selectbox("Policy", ["scripted", "minimind"])
    model_path = st.text_input("模型路径", "./minimind-3")
    max_steps = st.slider("最大步骤", 1, 30, 12)
    if st.button("运行 Agent", type="primary"):
        if policy_name == "minimind":
            policy = MiniMindPolicy(model_path)
        else:
            policy = ScriptedPolicy()
        with st.spinner("运行中"):
            result = AgentRuntime(
                policy, AgentLabConfig(max_steps=max_steps, runs_dir=ROOT / "runs")
            ).run(task)
        st.subheader("初始 Plan")
        st.json(result.trajectory.initial_plan)
        st.subheader("Tool Trace")
        for step in result.trajectory.steps:
            with st.expander(f"Step {step.step}: {step.action.get('kind')}", expanded=True):
                st.json(step.action)
                st.write("Observation:", step.observation_summary)
                st.write("Reflection:", step.reflection)
        left, right = st.columns(2)
        left.subheader("最终回答")
        left.write(result.state.final_answer or result.state.error)
        left.metric("总评分", f"{result.evaluation.score:.3f}")
        left.write("测试结果", result.trajectory.test_result)
        right.subheader("Git Diff")
        diff_path = Path(result.trajectory.trace_path).with_name("final_diff.patch")
        right.code(diff_path.read_text(encoding="utf-8"), language="diff")
        st.caption(f"Trajectory: {result.trajectory.trace_path}")


if __name__ == "__main__":
    main()
