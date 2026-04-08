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
import sys
import time
from typing import Any, List, Optional

from openai import OpenAI

from customer_support_env import CustomerSupportEnv
from customer_support_env.models import Action

# ── Environment variables ─────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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


# ── Tool schemas ──────────────────────────────────────────────────────────────

SUBMIT_TRIAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_triage",
        "description": "Submit the final triage decision for the support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "priority": {
                    "type": "string",
                    "enum": ["urgent", "high", "medium", "low"],
                    "description": "Urgency level of the ticket.",
                },
                "department": {
                    "type": "string",
                    "enum": ["billing", "technical", "account", "returns", "general"],
                    "description": "Department to route the ticket to.",
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["angry", "frustrated", "neutral", "satisfied"],
                    "description": "Detected customer sentiment.",
                },
                "escalate": {
                    "type": "boolean",
                    "description": "Whether the ticket requires manager escalation.",
                },
                "response_draft": {
                    "type": "string",
                    "description": "Draft reply to send to the customer.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for your decisions.",
                },
            },
            "required": ["priority"],
        },
    },
}

LOOKUP_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_history",
        "description": "Retrieve the customer's prior contact history.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

REQUEST_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": "request_info",
        "description": "Request additional context about the ticket (account status, order details, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "info_request": {
                    "type": "string",
                    "description": "What information you need.",
                }
            },
            "required": [],
        },
    },
}

ALL_TOOLS = [LOOKUP_HISTORY_TOOL, REQUEST_INFO_TOOL, SUBMIT_TRIAGE_TOOL]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _action_to_str(fn_name: str, fn_args: dict) -> str:
    """Compact single-line string for [STEP] action field."""
    if fn_name == "lookup_history":
        return "lookup_history()"
    if fn_name == "request_info":
        req = fn_args.get("info_request", "")
        req = req[:60].replace(" ", "_") if req else ""
        return f"request_info({req})"
    parts = []
    for key in ("priority", "department", "sentiment", "escalate"):
        val = fn_args.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return "submit_triage(" + ",".join(parts) + ")"


def _observation_to_messages(obs: Any) -> list[dict]:
    ticket = obs.ticket
    context_str = ""
    if obs.extra_context:
        context_str = "\n\nAdditional context revealed:\n" + json.dumps(obs.extra_context, indent=2)

    history_str = ""
    if obs.action_history:
        history_str = f"\n\nSteps taken: {len(obs.action_history)}/{obs.max_steps}"

    user_content = (
        f"=== SUPPORT TICKET ===\n"
        f"ID: {ticket.ticket_id}\n"
        f"Subject: {ticket.subject}\n"
        f"Body:\n{ticket.body}\n\n"
        f"Customer tier: {ticket.customer_tier}\n"
        f"Previous contacts (30d): {ticket.previous_contacts}\n"
        f"Account age: {ticket.account_age_days} days\n"
        f"Attachments: {', '.join(ticket.attachments) or 'none'}"
        f"{context_str}{history_str}"
    )
    return [
        {"role": "system", "content": obs.task_description},
        {"role": "user", "content": user_content},
    ]


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, env: CustomerSupportEnv, task_name: str) -> dict:
    """Run one episode, emit [START]/[STEP]/[END] logs, return result dict."""
    obs = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        messages = _observation_to_messages(obs)
        max_steps = env.task_config.max_steps

        for step in range(1, max_steps + 1):
            error_msg: Optional[str] = None
            action: Optional[Action] = None
            action_str = "no_op()"

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=ALL_TOOLS,
                    tool_choice="auto",
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    action = Action(action_type="submit_triage")
                    action_str = "submit_triage()"
                else:
                    tool_call = msg.tool_calls[0]
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    action_str = _action_to_str(fn_name, fn_args)

                    if fn_name == "lookup_history":
                        action = Action(action_type="lookup_history")
                    elif fn_name == "request_info":
                        action = Action(
                            action_type="request_info",
                            info_request=fn_args.get("info_request"),
                        )
                    else:
                        action = Action(
                            action_type="submit_triage",
                            priority=fn_args.get("priority"),
                            department=fn_args.get("department"),
                            sentiment=fn_args.get("sentiment"),
                            escalate=fn_args.get("escalate"),
                            response_draft=fn_args.get("response_draft"),
                            reasoning=fn_args.get("reasoning"),
                        )

            except Exception as exc:
                error_msg = str(exc)[:120]
                action = Action(action_type="submit_triage")
                action_str = "submit_triage()"

            obs, reward, done, info = env.step(action)
            rewards.append(reward.score)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward.score, done=done, error=error_msg)

            if done:
                score = reward.score
                break

            if action.action_type != "submit_triage" and msg and msg.tool_calls:
                tool_call = msg.tool_calls[0]
                messages = messages + [
                    msg.model_dump(exclude_unset=True),
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {"result": reward.feedback, "extra_context": obs.extra_context}
                        ),
                    },
                ]
                messages = _observation_to_messages(obs) + messages[-2:]

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "steps": steps_taken,
        "score": score,
        "success": success,
    }


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

    print("[INFO] Inference complete. Keeping container alive.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
