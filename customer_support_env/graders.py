"""
Deterministic graders for each task.

All graders return a (score, breakdown, feedback) triple where:
  score     – float in [0.0, 1.0]
  breakdown – dict mapping criterion → contribution to score
  feedback  – human-readable explanation
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .models import Action

# Priority adjacency — one level off earns partial credit
_PRIORITY_ORDER = ["urgent", "high", "medium", "low"]


def _priority_score(predicted: str, expected: str) -> float:
    if predicted == expected:
        return 1.0
    pi = _PRIORITY_ORDER.index(predicted)
    ei = _PRIORITY_ORDER.index(expected)
    if abs(pi - ei) == 1:
        return 0.5
    return 0.0


def _department_score(predicted: str, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def _sentiment_score(predicted: str, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def _escalate_score(predicted: bool, expected: bool) -> float:
    if predicted == expected:
        return 1.0
    # False negative on urgent escalation is worse than false positive
    if expected is True and predicted is False:
        return 0.0   # missed critical escalation
    return 0.3       # unnecessary escalation — mildly penalised


def _response_keyword_score(draft: str | None, keywords: list[str]) -> float:
    if not draft:
        return 0.0
    lower = draft.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return round(hits / len(keywords), 3) if keywords else 0.0


# ── Step-level penalty for unnecessary info-gathering ────────────────────────

def step_reward(action_type: str, ticket_extra_context: Dict[str, Any], step: int) -> float:
    """
    Small reward/penalty for non-terminal actions to provide trajectory signal.
      +0.05  lookup_history on a ticket that has a non-empty history
      +0.05  request_info  on a ticket that has non-empty extra_info
      -0.05  any lookup action after step 2 (diminishing returns)
    """
    if action_type == "lookup_history":
        has_history = bool(ticket_extra_context.get("_has_history"))
        base = 0.05 if has_history else -0.05
    elif action_type == "request_info":
        has_extra = bool(ticket_extra_context.get("_has_extra_info"))
        base = 0.05 if has_extra else -0.05
    else:
        base = 0.0

    if step > 2:
        base -= 0.05  # penalise late redundant lookups

    return round(max(-0.1, min(0.1, base)), 3)


# ── Task graders ─────────────────────────────────────────────────────────────

def grade_task1(action: Action, ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    """
    Task 1 – Priority Classification
    Weights: priority 100%
    """
    if action.priority is None:
        return 0.0, {"priority": 0.0}, "No priority submitted."

    p_score = _priority_score(action.priority, ground_truth["priority"])

    breakdown = {"priority": round(p_score, 3)}
    score = p_score
    feedback_parts = [f"Priority: {action.priority!r} vs expected {ground_truth['priority']!r} → {p_score:.1f}"]
    if p_score == 1.0:
        feedback_parts.append("Correct!")
    elif p_score == 0.5:
        feedback_parts.append("One level off — partial credit.")
    else:
        feedback_parts.append("Incorrect priority.")

    return round(score, 3), breakdown, " ".join(feedback_parts)


def grade_task2(action: Action, ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    """
    Task 2 – Ticket Routing
    Weights: priority 50%, department 50%
    """
    feedback_parts = []
    breakdown: Dict[str, float] = {}

    p_score = _priority_score(action.priority or "", ground_truth["priority"]) if action.priority else 0.0
    d_score = _department_score(action.department or "", ground_truth["department"]) if action.department else 0.0

    breakdown["priority"] = round(p_score * 0.5, 3)
    breakdown["department"] = round(d_score * 0.5, 3)

    feedback_parts.append(
        f"Priority: {action.priority!r} vs {ground_truth['priority']!r} → {p_score:.1f}"
    )
    feedback_parts.append(
        f"Department: {action.department!r} vs {ground_truth['department']!r} → {d_score:.1f}"
    )

    score = breakdown["priority"] + breakdown["department"]
    return round(score, 3), breakdown, " | ".join(feedback_parts)


def grade_task3(action: Action, ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    """
    Task 3 – Full Triage
    Weights: priority 0.25, department 0.20, sentiment 0.15, escalate 0.25, response 0.15
    Escalation is weighted highest because a missed escalation has real operational cost.
    """
    feedback_parts = []
    breakdown: Dict[str, float] = {}

    # Priority (0.25)
    p_score = _priority_score(action.priority or "", ground_truth["priority"]) if action.priority else 0.0
    breakdown["priority"] = round(p_score * 0.25, 3)
    feedback_parts.append(f"Priority({p_score:.1f})")

    # Department (0.20)
    d_score = _department_score(action.department or "", ground_truth["department"]) if action.department else 0.0
    breakdown["department"] = round(d_score * 0.20, 3)
    feedback_parts.append(f"Department({d_score:.1f})")

    # Sentiment (0.15)
    s_score = _sentiment_score(action.sentiment or "", ground_truth["sentiment"]) if action.sentiment else 0.0
    breakdown["sentiment"] = round(s_score * 0.15, 3)
    feedback_parts.append(f"Sentiment({s_score:.1f})")

    # Escalation (0.25)
    e_score = _escalate_score(
        action.escalate if action.escalate is not None else False,
        ground_truth["escalate"],
    )
    breakdown["escalate"] = round(e_score * 0.25, 3)
    feedback_parts.append(f"Escalate({e_score:.1f})")

    # Response quality (0.15)
    keywords = ground_truth.get("response_keywords", [])
    r_score = _response_keyword_score(action.response_draft, keywords)
    breakdown["response"] = round(r_score * 0.15, 3)
    feedback_parts.append(f"Response({r_score:.2f})")

    score = sum(breakdown.values())
    return round(score, 3), breakdown, " | ".join(feedback_parts)


# ── Dispatch ─────────────────────────────────────────────────────────────────

GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
