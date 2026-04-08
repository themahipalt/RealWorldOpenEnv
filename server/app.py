"""
server/app.py — OpenEnv HTTP server entry point

Exposes CustomerSupportEnv over HTTP for automated validation:
  GET  /        → 200 health check
  POST /reset   → env.reset()
  POST /step    → env.step(action)
  GET  /state   → env.state()
  GET  /tasks   → list all tasks
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel

# Allow imports from repo root when run directly from server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from customer_support_env import CustomerSupportEnv
from customer_support_env.models import Action
from customer_support_env.tasks import TASK_CONFIGS

app = FastAPI(title="CustomerSupportEnv", version="1.0.0")

_envs: dict[str, CustomerSupportEnv] = {}


class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: int = 42


class StepRequest(BaseModel):
    task_id: str = "task1"
    action_type: str
    priority: Optional[str] = None
    department: Optional[str] = None
    sentiment: Optional[str] = None
    escalate: Optional[bool] = None
    response_draft: Optional[str] = None
    info_request: Optional[str] = None
    reasoning: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "environment": "customer-support-triage", "version": "1.0.0"}


from fastapi import Body
from typing import Optional

@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()

    env = CustomerSupportEnv(task_id=req.task_id, seed=req.seed)
    _envs[req.task_id] = env

    obs = env.reset()

    return {
        "observation": obs.model_dump(),
        "reward": 0,
        "done": False,
        "info": {}
    }


@app.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.task_id)
    if env is None:
        return {"error": f"No active episode for task_id={req.task_id!r}. Call /reset first."}
    action = Action(
        action_type=req.action_type,
        priority=req.priority,
        department=req.department,
        sentiment=req.sentiment,
        escalate=req.escalate,
        response_draft=req.response_draft,
        info_request=req.info_request,
        reasoning=req.reasoning,
    )
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = "task1"):
    env = _envs.get(task_id)
    if env is None:
        return {"error": f"No active episode for task_id={task_id!r}. Call /reset first."}
    return env.state()


@app.get("/tasks")
def tasks():
    return {
        k: {
            "name": v.name,
            "difficulty": v.difficulty,
            "description": v.description,
            "max_steps": v.max_steps,
            "reward_threshold": v.reward_threshold,
            "required_fields": v.required_fields,
        }
        for k, v in TASK_CONFIGS.items()
    }


def _run_inference():
    root = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run([sys.executable, os.path.join(root, "inference.py")], cwd=root)


@app.on_event("startup")
def startup_event():
    threading.Thread(target=_run_inference, daemon=True).start()


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
