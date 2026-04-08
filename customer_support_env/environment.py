"""
CustomerSupportEnv — OpenEnv-compliant environment for customer support ticket triage.

Interface:
  reset()         → Observation
  step(action)    → (Observation, Reward, done: bool, info: dict)
  state()         → dict
"""
from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, Tuple

from .graders import GRADERS, step_reward, _safe_score
from .models import Action, Observation, Reward, Ticket
from .tasks import TASK_CONFIGS
from .tickets import TICKET_POOL


class CustomerSupportEnv:
    """
    A multi-step environment for customer support ticket triage.

    Each episode:
      1. reset()  picks a random ticket from the task's pool.
      2. Agent calls step() with lookup_history / request_info to gather context.
      3. Agent calls step() with submit_triage to end the episode and receive
         the final graded reward.

    Intermediate steps provide small positive/negative rewards to guide the
    agent's information-gathering strategy (trajectory-level signal).
    """

    def __init__(self, task_id: str, seed: int | None = None) -> None:
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id {task_id!r}. Choose from {list(TASK_CONFIGS)}")
        self.task_id = task_id
        self.task_config = TASK_CONFIGS[task_id]
        self._rng = random.Random(seed)

        # Episode state — initialised by reset()
        self._ticket_record: Dict[str, Any] = {}
        self._step: int = 0
        self._done: bool = False
        self._extra_context: Dict[str, Any] = {}
        self._action_history: list[Dict[str, Any]] = []
        self._cumulative_score: float = 0.0

    # ── Public OpenEnv interface ──────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode with a randomly chosen ticket."""
        pool = TICKET_POOL[self.task_id]
        self._ticket_record = deepcopy(self._rng.choice(pool))

        self._step = 0
        self._done = False
        self._extra_context = {
            # Hints used by step_reward() — not shown to agent
            "_has_history": bool(self._ticket_record["history"].get("last_contacts")),
            "_has_extra_info": bool(self._ticket_record.get("extra_info")),
        }
        self._action_history = []
        self._cumulative_score = 0.0

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one agent action.

        Returns
        -------
        observation : Observation
        reward      : Reward
        done        : bool
        info        : dict  (step count, cumulative score, etc.)
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new one.")

        self._step += 1

        if action.action_type == "lookup_history":
            reward = self._handle_lookup_history()
        elif action.action_type == "request_info":
            reward = self._handle_request_info(action)
        elif action.action_type == "submit_triage":
            reward = self._handle_submit_triage(action)
        else:
            reward = Reward(
                score=0.0,
                breakdown={},
                feedback=f"Unknown action_type: {action.action_type!r}",
                is_terminal=False,
            )

        # Force-terminate if max steps exceeded without submission
        if self._step >= self.task_config.max_steps and not self._done:
            self._done = True
            reward = Reward(
                score=0.0,
                breakdown={"timeout": 0.0},
                feedback=(
                    f"Max steps ({self.task_config.max_steps}) reached without "
                    "submitting triage. Episode terminated."
                ),
                is_terminal=True,
            )

        # Ensure score is strictly within (0, 1) — never exactly 0.0 or 1.0
        reward = reward.model_copy(update={"score": _safe_score(reward.score)})

        self._cumulative_score += reward.score
        self._action_history.append(
            {"step": self._step, "action_type": action.action_type, "score": reward.score}
        )

        info = {
            "step": self._step,
            "max_steps": self.task_config.max_steps,
            "cumulative_score": round(self._cumulative_score, 3),
            "task_id": self.task_id,
        }
        return self._make_observation(), reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return a snapshot of the full internal episode state."""
        return {
            "task_id": self.task_id,
            "ticket_id": self._ticket_record.get("ticket", {}).get("ticket_id"),
            "step": self._step,
            "max_steps": self.task_config.max_steps,
            "done": self._done,
            "cumulative_score": round(self._cumulative_score, 3),
            "extra_context_revealed": {
                k: v for k, v in self._extra_context.items() if not k.startswith("_")
            },
            "action_history": self._action_history,
            "ground_truth": self._ticket_record.get("ground_truth"),
        }

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_lookup_history(self) -> Reward:
        if "history" in self._extra_context:
            return Reward(
                score=0.0,
                breakdown={"lookup_history": 0.0},
                feedback="History already revealed — no additional information.",
                is_terminal=False,
            )

        self._extra_context["history"] = self._ticket_record["history"]
        raw_score = step_reward("lookup_history", self._extra_context, self._step)
        return Reward(
            score=max(0.0, raw_score),   # non-negative for observation
            breakdown={"lookup_history": max(0.0, raw_score)},
            feedback=(
                "Customer history revealed. "
                + (
                    "Useful context found."
                    if self._extra_context["_has_history"]
                    else "No prior contacts — lookup was unnecessary."
                )
            ),
            is_terminal=False,
        )

    def _handle_request_info(self, action: Action) -> Reward:
        if "extra_info" in self._extra_context:
            return Reward(
                score=0.0,
                breakdown={"request_info": 0.0},
                feedback="Extra info already revealed — no additional information.",
                is_terminal=False,
            )

        self._extra_context["extra_info"] = self._ticket_record.get("extra_info", {})
        raw_score = step_reward("request_info", self._extra_context, self._step)
        return Reward(
            score=max(0.0, raw_score),
            breakdown={"request_info": max(0.0, raw_score)},
            feedback=(
                "Additional context revealed. "
                + (
                    "Relevant details found."
                    if self._extra_context["_has_extra_info"]
                    else "No extra info available — request was unnecessary."
                )
            ),
            is_terminal=False,
        )

    def _handle_submit_triage(self, action: Action) -> Reward:
        self._done = True
        ground_truth = self._ticket_record["ground_truth"]
        grader = GRADERS[self.task_id]
        score, breakdown, feedback = grader(action, ground_truth)
        return Reward(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            is_terminal=True,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        ticket_data = self._ticket_record.get("ticket", {})
        ticket = Ticket(**ticket_data)

        # Only expose non-private context keys to the agent
        visible_context = {
            k: v for k, v in self._extra_context.items() if not k.startswith("_")
        }

        return Observation(
            ticket=ticket,
            task_id=self.task_id,
            task_description=self.task_config.instructions,
            step=self._step,
            max_steps=self.task_config.max_steps,
            extra_context=visible_context,
            action_history=list(self._action_history),
        )
