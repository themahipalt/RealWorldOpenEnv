"""
inference.py — CustomerSupportEnv inference script

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face API key (no default).
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, List, Optional

from openai import OpenAI

from customer_support_env import CustomerSupportEnv
from customer_support_env.models import Action

# ── Environment variables ─────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "customer-support-triage"
EPISODES_PER_TASK = 3
SEED = 42
SUCCESS_SCORE_THRESHOLD = 0.6

# ── Structured logging ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_prompt(obs: Any) -> str:
    ticket = obs.ticket
    task_id = obs.task_id

    ticket_block = (
        f"=== SUPPORT TICKET ===\n"
        f"Subject: {ticket.subject}\n"
        f"Body: {ticket.body}\n"
        f"Customer tier: {ticket.customer_tier}\n"
        f"Previous contacts (30d): {ticket.previous_contacts}\n"
        f"Account age: {ticket.account_age_days} days\n"
    )

    if task_id == "task1":
        schema = '{"priority": "urgent|high|medium|low"}'
        instructions = (
            "Classify the ticket urgency. "
            "urgent=financial loss/security/outage, high=major issue no workaround, "
            "medium=moderate issue workaround exists, low=minor/general."
        )
    elif task_id == "task2":
        schema = '{"priority": "urgent|high|medium|low", "department": "billing|technical|account|returns|general"}'
        instructions = (
            "Classify priority AND route to department. "
            "billing=payments/invoices, technical=bugs/API, account=login/subscription, "
            "returns=refunds/replacements, general=how-to/feature requests."
        )
    else:  # task3
        schema = (
            '{"priority": "urgent|high|medium|low", '
            '"department": "billing|technical|account|returns|general", '
            '"sentiment": "angry|frustrated|neutral|satisfied", '
            '"escalate": true|false, '
            '"response_draft": "short professional reply to customer"}'
        )
        instructions = (
            "Full triage: priority, department, sentiment, escalation decision, and a draft reply. "
            "Escalate if: 3+ unresolved contacts, financial loss, security risk, cancellation threat, or production outage."
        )

    return (
        f"{obs.task_description}\n\n"
        f"{ticket_block}\n"
        f"{instructions}\n\n"
        f"Respond with ONLY valid JSON matching this schema:\n{schema}"
    )


def _parse_response(text: str, task_id: str) -> Action:
    """Extract JSON from model response and map to Action."""
    data: dict = {}
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    escalate = data.get("escalate")
    if isinstance(escalate, str):
        escalate = escalate.lower() == "true"

    return Action(
        action_type="submit_triage",
        priority=data.get("priority"),
        department=data.get("department"),
        sentiment=data.get("sentiment"),
        escalate=bool(escalate) if escalate is not None else None,
        response_draft=data.get("response_draft"),
    )


def _action_to_str(action: Action) -> str:
    parts = []
    for key in ("priority", "department", "sentiment", "escalate"):
        val = getattr(action, key, None)
        if val is not None:
            parts.append(f"{key}={val}")
    return "submit_triage(" + ",".join(parts) + ")"


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, env: CustomerSupportEnv, task_name: str) -> dict:
    """Run one episode, emit [START]/[STEP]/[END] logs, return result dict."""
    obs = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        prompt = _build_prompt(obs)
        error_msg: Optional[str] = None
        action_str = "submit_triage()"
        action = Action(action_type="submit_triage")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1,
            )
            text = (response.choices[0].message.content or "").strip()
            action = _parse_response(text, obs.task_id)
            action_str = _action_to_str(action)
        except Exception as exc:
            error_msg = str(exc)[:120]

        obs, reward, done, info = env.step(action)
        rewards.append(reward.score)
        steps_taken = 1
        score = reward.score

        log_step(step=1, action=action_str, reward=reward.score, done=done, error=error_msg)

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "steps": steps_taken, "score": score, "success": success}


# ── Main ──────────────────────────────────────────────────────────────────────

TASKS = [
    ("task1", "priority-classification"),
    ("task2", "ticket-routing"),
    ("task3", "full-triage"),
]


def main() -> None:
    if not HF_TOKEN:
        sys.exit("Error: HF_TOKEN environment variable not set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id, task_name in TASKS:
        env = CustomerSupportEnv(task_id=task_id, seed=SEED)
        for ep in range(1, EPISODES_PER_TASK + 1):
            run_episode(client, env, task_name)

    print("[INFO] Inference complete.", flush=True)


if __name__ == "__main__":
    main()
