"""
Baseline inference script — runs an OpenAI model against all three tasks
and reports per-task and aggregate scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py [--model gpt-4o-mini] [--episodes 5] [--seed 42]

Requirements:
    pip install openai pydantic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any

from openai import OpenAI

from customer_support_env import CustomerSupportEnv
from customer_support_env.models import Action

# ── OpenAI function / tool schema derived from Action model ──────────────────

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


def _observation_to_messages(obs: Any) -> list[dict]:
    """Convert an Observation to an OpenAI messages list."""
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


def run_episode(client: OpenAI, model: str, env: CustomerSupportEnv) -> dict:
    """Run one episode and return result dict."""
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    final_score = 0.0

    messages = _observation_to_messages(obs)

    while not done:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=ALL_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Model responded with text instead of tool — force submit with no fields
            action = Action(action_type="submit_triage")
            obs, reward, done, info = env.step(action)
            final_score = reward.score
            break

        tool_call = msg.tool_calls[0]
        fn_name = tool_call.function.name
        try:
            fn_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            fn_args = {}

        # Build Action
        if fn_name == "lookup_history":
            action = Action(action_type="lookup_history")
        elif fn_name == "request_info":
            action = Action(action_type="request_info", info_request=fn_args.get("info_request"))
        else:  # submit_triage
            action = Action(
                action_type="submit_triage",
                priority=fn_args.get("priority"),
                department=fn_args.get("department"),
                sentiment=fn_args.get("sentiment"),
                escalate=fn_args.get("escalate"),
                response_draft=fn_args.get("response_draft"),
                reasoning=fn_args.get("reasoning"),
            )

        obs, reward, done, info = env.step(action)
        total_reward += reward.score
        steps += 1

        if reward.is_terminal:
            final_score = reward.score
            break

        # Feed tool result back to model for next step
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
        # Refresh observation into messages
        messages = _observation_to_messages(obs) + messages[-2:]

    return {
        "ticket_id": env.state().get("ticket_id"),
        "steps": steps,
        "final_score": round(final_score, 3),
        "total_reward": round(total_reward, 3),
        "passed": final_score >= env.task_config.reward_threshold,
    }


def run_task(client: OpenAI, model: str, task_id: str, episodes: int, seed: int) -> dict:
    env = CustomerSupportEnv(task_id=task_id, seed=seed)
    results = []
    print(f"\n{'─'*60}")
    print(f"Task: {env.task_config.name}  ({env.task_config.difficulty})  [{episodes} episodes]")
    print(f"{'─'*60}")

    for ep in range(episodes):
        result = run_episode(client, model, env)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  ep {ep+1:02d}  ticket={result['ticket_id']}  "
            f"score={result['final_score']:.3f}  steps={result['steps']}  [{status}]"
        )

    scores = [r["final_score"] for r in results]
    pass_rate = mean(r["passed"] for r in results)
    summary = {
        "task_id": task_id,
        "difficulty": env.task_config.difficulty,
        "mean_score": round(mean(scores), 3),
        "min_score": round(min(scores), 3),
        "max_score": round(max(scores), 3),
        "pass_rate": round(pass_rate, 3),
        "reward_threshold": env.task_config.reward_threshold,
    }
    print(
        f"\n  Summary → mean={summary['mean_score']}  "
        f"pass_rate={summary['pass_rate']:.0%}  "
        f"(threshold={summary['reward_threshold']})"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for CustomerSupportEnv")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model ID")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    print(f"\nBaseline: {args.model}  |  {args.episodes} episodes/task  |  seed={args.seed}")

    all_summaries = []
    for task_id in ["task1", "task2", "task3"]:
        summary = run_task(client, args.model, task_id, args.episodes, args.seed)
        all_summaries.append(summary)

    overall_mean = mean(s["mean_score"] for s in all_summaries)
    overall_pass = mean(s["pass_rate"] for s in all_summaries)

    print(f"\n{'═'*60}")
    print(f"OVERALL  mean_score={overall_mean:.3f}   pass_rate={overall_pass:.0%}")
    print(f"{'═'*60}\n")

    print("Per-task results (JSON):")
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
