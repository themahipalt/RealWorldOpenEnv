# Customer Support Triage — OpenEnv Environment

A real-world agent benchmark where models must triage customer support tickets:
classify urgency, route to departments, detect sentiment, decide on escalation,
and draft replies.

---

## Motivation

Customer support triage is a high-volume, high-stakes daily task at any product
company. Poor triage causes SLA breaches, customer churn, and wasted engineering
time. This environment tests whether an LLM agent can replicate the judgment of an
experienced Tier-1 support analyst.

---

## Action Space

| Field | Type | Values |
|-------|------|--------|
| `action_type` | string (required) | `submit_triage`, `lookup_history`, `request_info` |
| `priority` | string | `urgent`, `high`, `medium`, `low` |
| `department` | string | `billing`, `technical`, `account`, `returns`, `general` |
| `sentiment` | string | `angry`, `frustrated`, `neutral`, `satisfied` |
| `escalate` | bool | `true` / `false` |
| `response_draft` | string | Free-form reply draft |
| `info_request` | string | What additional info is needed |

**Non-terminal actions** (`lookup_history`, `request_info`) reveal additional
context at a small reward cost and must be used before `submit_triage`.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket` | Ticket | The support ticket (id, subject, body, tier, age, …) |
| `task_id` | string | Which task is active |
| `task_description` | string | Full agent instructions |
| `step` | int | Current step (0-indexed) |
| `max_steps` | int | Episode step limit |
| `extra_context` | dict | Context revealed by lookup/request actions |
| `action_history` | list | Previous (action_type, score) pairs |

---

## Tasks

| ID | Name | Difficulty | Required fields | Threshold |
|----|------|-----------|-----------------|-----------|
| `task1` | Priority Classification | Easy | `priority` | 0.80 |
| `task2` | Ticket Routing | Medium | `priority`, `department` | 0.70 |
| `task3` | Full Triage | Hard | all fields + `response_draft` | 0.60 |

### Reward breakdown

**Task 1**: `priority` accuracy (1.0 exact, 0.5 one-level-off, 0.0 otherwise).

**Task 2**: `priority` × 0.5 + `department` × 0.5.

**Task 3**: `priority` × 0.25 + `department` × 0.20 + `sentiment` × 0.15 +
`escalate` × 0.25 + `response_draft` quality × 0.15.

Intermediate steps yield ±0.05 based on whether information gathering was
relevant to the ticket's content, providing trajectory-level signal.

---

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python baseline.py --model gpt-4o-mini --episodes 5
```

### Docker

```bash
docker build -t cst-env .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY cst-env
```

### Programmatic usage

```python
from customer_support_env import CustomerSupportEnv
from customer_support_env.models import Action

env = CustomerSupportEnv(task_id="task2", seed=42)
obs = env.reset()
print(obs.ticket.subject)

# Optional context gathering
_, reward, done, info = env.step(Action(action_type="lookup_history"))

# Submit triage
action = Action(
    action_type="submit_triage",
    priority="high",
    department="technical",
)
obs, reward, done, info = env.step(action)
print(reward.score, reward.feedback)
```

---

## Baseline Scores (gpt-4o-mini, 5 episodes, seed=42)

| Task | Mean Score | Pass Rate |
|------|-----------|-----------|
| task1 (easy) | ~0.85 | ~100% |
| task2 (medium) | ~0.70 | ~80% |
| task3 (hard) | ~0.55 | ~60% |

*Run `python baseline.py` to reproduce.*

---

## Project structure

```
customer_support_env/
  __init__.py      — package exports
  models.py        — Observation, Action, Reward (Pydantic)
  environment.py   — CustomerSupportEnv (reset / step / state)
  tasks.py         — TaskConfig definitions
  graders.py       — deterministic per-task graders
  tickets.py       — curated ticket corpus with ground truth
baseline.py        — OpenAI inference script
openenv.yaml       — OpenEnv spec metadata
Dockerfile
requirements.txt
```
