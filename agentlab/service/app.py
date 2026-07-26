from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentlab.config import AgentLabConfig
from agentlab.evaluation import get_task, load_tasks, write_report
from agentlab.policies import MiniMindPolicy, ScriptedPolicy
from agentlab.runtime import AgentRuntime
from agentlab.service.run_store import RunStore


ROOT = Path(__file__).resolve().parents[2]
store = RunStore(ROOT / "runs" / "agentlab.sqlite3")
app = FastAPI(title="MiniMind-AgentLab", version="0.1.0")


class RunRequest(BaseModel):
    task_id: str
    policy: Literal["scripted", "minimind"] = "scripted"
    model_path: str | None = None
    max_steps: int = Field(default=12, ge=1, le=50)


class EvaluationRequest(BaseModel):
    split: Literal["train", "dev", "test"] = "test"
    policy: Literal["scripted", "minimind"] = "scripted"
    model_path: str | None = None
    max_steps: int = Field(default=12, ge=1, le=50)


def policy_from_request(policy: str, model_path: str | None):
    if policy == "scripted":
        return ScriptedPolicy()
    if not model_path:
        raise HTTPException(422, "model_path is required for minimind policy")
    return MiniMindPolicy(model_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/tasks")
def tasks() -> list[dict]:
    return [
        {
            "task_id": task.task_id,
            "repo_id": task.repo_id,
            "task_type": task.task_type,
            "split": task.split,
            "prompt": task.user_query,
        }
        for task in load_tasks()
    ]


@app.post("/v1/agent/runs")
def create_run(request: RunRequest) -> dict:
    try:
        task = get_task(request.task_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    result = AgentRuntime(
        policy_from_request(request.policy, request.model_path),
        AgentLabConfig(max_steps=request.max_steps, runs_dir=ROOT / "runs"),
    ).run(task)
    payload = {
        "run_id": result.trajectory.run_id,
        "task_id": task.task_id,
        "status": result.state.status.value,
        "success": result.evaluation.success,
        "score": result.evaluation.score,
        "trace_path": result.trajectory.trace_path,
    }
    store.put(payload)
    return payload


@app.get("/v1/agent/runs/{run_id}")
def get_run(run_id: str) -> dict:
    payload = store.get(run_id)
    if not payload:
        raise HTTPException(404, "run not found")
    return payload


@app.get("/v1/agent/runs/{run_id}/trace")
def get_trace(run_id: str) -> dict:
    payload = store.get(run_id)
    if not payload:
        raise HTTPException(404, "run not found")
    path = Path(payload["trace_path"])
    if not path.is_file():
        raise HTTPException(404, "trace file not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/v1/evaluations")
def create_evaluation(request: EvaluationRequest) -> dict:
    policy = policy_from_request(request.policy, request.model_path)
    results = [
        AgentRuntime(
            policy, AgentLabConfig(max_steps=request.max_steps, runs_dir=ROOT / "runs")
        ).run(task).evaluation
        for task in load_tasks(request.split)
    ]
    json_path, markdown_path = write_report(results, ROOT / "reports")
    return {
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
        "successes": sum(item.success for item in results),
        "tasks": len(results),
    }
