"""
Task definitions — metadata and per-task configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskConfig:
    task_id: str
    name: str
    difficulty: str          # "easy" | "medium" | "hard"
    description: str
    instructions: str        # shown to agent in task_description
    max_steps: int
    required_fields: List[str]   # fields that must be set on submit_triage
    reward_threshold: float  # score considered a "pass"


TASK_CONFIGS: dict[str, TaskConfig] = {
    "task1": TaskConfig(
        task_id="task1",
        name="Priority Classification",
        difficulty="easy",
        description=(
            "Read a customer support ticket and classify its urgency as one of: "
            "urgent, high, medium, low."
        ),
        instructions=(
            "You are a customer support triage agent.\n"
            "Your ONLY job is to classify the priority of the incoming ticket.\n"
            "Priority levels:\n"
            "  urgent – immediate risk (financial loss, security breach, production outage)\n"
            "  high   – significant user impact, no workaround\n"
            "  medium – moderate impact, workaround exists\n"
            "  low    – minor issue or general enquiry\n\n"
            "Submit your answer using action_type='submit_triage' with the 'priority' field set."
        ),
        max_steps=3,
        required_fields=["priority"],
        reward_threshold=0.8,
    ),

    "task2": TaskConfig(
        task_id="task2",
        name="Ticket Routing",
        difficulty="medium",
        description=(
            "Read a customer support ticket, classify its priority, and route it "
            "to the correct department: billing, technical, account, returns, general."
        ),
        instructions=(
            "You are a customer support triage agent.\n"
            "Your job is to classify the ticket priority AND route it to the correct department.\n"
            "Priority levels: urgent / high / medium / low\n"
            "Departments:\n"
            "  billing   – payment issues, invoices, charges\n"
            "  technical – bugs, errors, API, integrations\n"
            "  account   – login, password, permissions, subscription access\n"
            "  returns   – product returns, refunds, replacements\n"
            "  general   – how-to questions, feature requests, general enquiries\n\n"
            "You may call lookup_history or request_info once each before submitting.\n"
            "Submit using action_type='submit_triage' with 'priority' and 'department' set."
        ),
        max_steps=4,
        required_fields=["priority", "department"],
        reward_threshold=0.7,
    ),

    "task3": TaskConfig(
        task_id="task3",
        name="Full Triage",
        difficulty="hard",
        description=(
            "Perform a complete ticket triage: priority, department, customer sentiment, "
            "escalation decision, and a draft reply to the customer."
        ),
        instructions=(
            "You are a senior customer support triage agent.\n"
            "Perform a full triage of the incoming ticket:\n"
            "  priority       – urgent / high / medium / low\n"
            "  department     – billing / technical / account / returns / general\n"
            "  sentiment      – angry / frustrated / neutral / satisfied\n"
            "  escalate       – true if the issue requires manager/senior intervention\n"
            "  response_draft – a short, professional reply to send to the customer\n\n"
            "Escalation criteria:\n"
            "  - Customer has made 3+ unresolved contacts on the same issue\n"
            "  - Financial loss or security risk is present\n"
            "  - Customer has explicitly threatened to cancel or dispute charges\n"
            "  - Production outage affecting multiple users\n\n"
            "You may call lookup_history and/or request_info before submitting.\n"
            "Submit using action_type='submit_triage' with all fields set."
        ),
        max_steps=5,
        required_fields=["priority", "department", "sentiment", "escalate", "response_draft"],
        reward_threshold=0.6,
    ),
}
