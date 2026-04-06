"""
Typed Pydantic models for the Customer Support Triage OpenEnv environment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ── Shared literal types ──────────────────────────────────────────────────────

Priority   = Literal["urgent", "high", "medium", "low"]
Department = Literal["billing", "technical", "account", "returns", "general"]
Sentiment  = Literal["angry", "frustrated", "neutral", "satisfied"]
ActionType = Literal["submit_triage", "request_info", "lookup_history"]


# ── Core domain object ────────────────────────────────────────────────────────

class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["premium", "standard", "basic"]
    previous_contacts: int = Field(ge=0, description="Number of prior contacts in last 30 days")
    account_age_days: int = Field(ge=0)
    attachments: List[str] = []


# ── OpenEnv: Observation ──────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent receives at every step."""

    ticket: Ticket
    task_id: str
    task_description: str
    step: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    # Accumulates revealed context after lookup/request actions
    extra_context: Dict[str, Any] = {}
    # Log of (action_type, result) pairs seen so far
    action_history: List[Dict[str, Any]] = []


# ── OpenEnv: Action ───────────────────────────────────────────────────────────

class Action(BaseModel):
    """
    Three action types:
      - lookup_history   : reveals past interaction history for this customer
      - request_info     : requests clarifying details (product/account status)
      - submit_triage    : final answer — ends the episode
    """

    action_type: ActionType

    # submit_triage fields
    priority: Optional[Priority] = None
    department: Optional[Department] = None
    sentiment: Optional[Sentiment] = None
    escalate: Optional[bool] = None
    response_draft: Optional[str] = Field(
        default=None,
        description="Draft reply to send to the customer (task 3 only)",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's chain-of-thought (not scored, kept for logging)",
    )

    # request_info field
    info_request: Optional[str] = Field(
        default=None,
        description="What additional information is being requested",
    )


# ── OpenEnv: Reward ───────────────────────────────────────────────────────────

class Reward(BaseModel):
    """Reward signal returned after every step."""

    score: float = Field(ge=0.0, le=1.0, description="Normalised score for this step")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion scores that sum to `score`",
    )
    feedback: str = Field(description="Human-readable explanation of the score")
    is_terminal: bool = Field(description="True when the episode has ended")
