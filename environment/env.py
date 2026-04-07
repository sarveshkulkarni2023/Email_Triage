"""
EmailTriageEnv — the core OpenEnv environment.

Implements the standard interface:
    reset(task_id, seed) → Observation
    step(action)         → (observation, reward, done, info)
    state()              → current internal state dict
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from environment.constants import (
    Difficulty,
    Stage,
    PENALTY_PER_EXTRA_STEP,
    PENALTY_UNNECESSARY_STEP,
    REWARD_WEIGHTS,
)
from environment.state_manager import StateManager
from graders.base_grader import BaseGrader
from graders.llm_grader import LLMGrader
from graders.rule_based import RuleBasedGrader
from models.action import Action
from models.observation import Observation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Task registry
# ------------------------------------------------------------------ #

_TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy": {
        "difficulty": Difficulty.EASY,
        "data_path": "data/easy_emails.json",
        "grader_cls": RuleBasedGrader,
    },
    "medium": {
        "difficulty": Difficulty.MEDIUM,
        "data_path": "data/medium_emails.json",
        "grader_cls": RuleBasedGrader,
    },
    "hard": {
        "difficulty": Difficulty.HARD,
        "data_path": "data/hard_emails.json",
        "grader_cls": LLMGrader,
    },
}


class EmailTriageEnv:
    """OpenEnv-compliant email-triage environment."""

    def __init__(self) -> None:
        self._state_manager = StateManager()
        self._grader: Optional[BaseGrader] = None
        self._episode_rewards: list[float] = []

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        task_id : str
            One of ``"easy"``, ``"medium"``, ``"hard"``.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        dict
            JSON-serialisable observation.
        """
        if task_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose from {list(_TASK_REGISTRY)}"
            )

        task_cfg = _TASK_REGISTRY[task_id]
        self._grader = task_cfg["grader_cls"]()
        self._episode_rewards = []

        obs: Observation = self._state_manager.reset(
            task_id=task_id,
            difficulty=task_cfg["difficulty"],
            data_path=task_cfg["data_path"],
            seed=seed,
        )
        logger.info("Environment reset — task=%s seed=%s", task_id, seed)
        return obs.model_dump()

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #

    def step(
        self, action_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self._state_manager.done:
            raise RuntimeError("Episode has ended. Call reset() first.")

        action = Action(**action_dict)
        self._state_manager.record_action(action_dict)
        assert self._grader is not None

        stage = self._state_manager.current_stage
        truth = self._state_manager.ground_truth
        diff = self._state_manager.difficulty
        weights = REWARD_WEIGHTS[diff]
        email_body = self._state_manager.observation.email.body

        reward_earned = 0.0
        penalty = 0.0
        advanced = False
        details = f"Processed stage: {stage.value}"

        if stage == Stage.CLASSIFICATION:
            if action.classification is None:
                penalty += 0.1
                details = "Missing classification in action."
            else:
                reward_earned = self._grader.grade_classification(action, truth, weights["classification"])
                advanced = True

        elif stage == Stage.PRIORITY:
            if action.priority is None:
                penalty += 0.1
                details = "Missing priority in action."
            else:
                reward_earned = self._grader.grade_priority(action, truth, weights["priority"])
                advanced = True

        elif stage == Stage.ACTION:
            if action.action is None:
                penalty += 0.1
                details = "Missing action in action."
            else:
                reward_earned = self._grader.grade_action(action, truth, weights["action"])
                advanced = True

        elif stage == Stage.RESPONSE:
            if not action.response_text:
                penalty += 0.1
                details = "Missing response_text in action."
            else:
                reward_earned = self._grader.grade_response(action, truth, email_body, weights["response_quality"])
                advanced = True

        # Apply per-step penalty to discourage loops
        penalty += PENALTY_PER_EXTRA_STEP

        incremental_reward = max(-1.0, reward_earned - penalty)
        self._state_manager.add_reward(incremental_reward)
        self._episode_rewards.append(incremental_reward)
        self._state_manager.set_feedback(details)

        self._state_manager.advance_step()
        if advanced:
            self._state_manager.advance_stage()

        done = self._state_manager.done
        obs = (
            self._state_manager.observation.model_dump()
            if not done
            else self._terminal_observation()
        )

        info = {
            "incremental_reward": round(incremental_reward, 4),
            "stage_advanced": advanced,
            "accumulated_reward": round(self._state_manager.accumulated_reward, 4),
            "episode_rewards": [round(r, 4) for r in self._episode_rewards],
            "details": details
        }

        logger.info(
            "step=%d reward=%.4f done=%s",
            self._state_manager.step_number,
            incremental_reward,
            done,
        )

        return obs, round(incremental_reward, 4), done, info

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #

    def state(self) -> Dict[str, Any]:
        """Return a snapshot of the internal state (no ground truth)."""
        return self._state_manager.state_snapshot()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _terminal_observation(self) -> Dict[str, Any]:
        """Build a terminal observation dict."""
        return {
            "step": self._state_manager.step_number,
            "email": None,
            "task_id": self._state_manager.task_id,
            "difficulty": self._state_manager.difficulty.value,
            "remaining_steps": 0,
            "feedback": "Episode complete.",
            "terminal": True,
            "accumulated_reward": round(self._state_manager.accumulated_reward, 4),
        }
