"""
State manager — manages episode lifecycle and internal state.

Responsibilities:
  • Load task data from JSON datasets.
  • Track episode step counters, accumulated rewards.
  • Provide the current observation.
  • Enforce max-step termination.
  • Prevent state leakage between episodes.
"""

from __future__ import annotations

import json
import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment.constants import (
    MAX_STEPS_PER_EPISODE,
    Difficulty,
    Stage,
)
from models.observation import EmailObservation, Observation


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class StateManager:
    """Encapsulates all mutable state for a single episode."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self._emails: List[Dict[str, Any]] = []
        self._current_email_index: int = 0
        self._current_email: Optional[Dict[str, Any]] = None
        self._ground_truth: Optional[Dict[str, Any]] = None
        self._step: int = 0
        self._max_steps: int = MAX_STEPS_PER_EPISODE
        self._done: bool = True
        self._difficulty: Difficulty = Difficulty.EASY
        self._task_id: str = ""
        self._accumulated_reward: float = 0.0
        self._current_stage: Stage = Stage.CLASSIFICATION
        self._customer_sentiment: str = "neutral"
        self._action_history: List[Dict[str, Any]] = []
        self._feedback: Optional[str] = None
        self._seed: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_dataset(path: str) -> List[Dict[str, Any]]:
        """Load a JSON dataset from *path* (relative to project root)."""
        full_path = Path(__file__).resolve().parent.parent / path
        if not full_path.exists():
            raise FileNotFoundError(f"Dataset not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Dataset must be a non-empty JSON array: {full_path}")
        return data

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #

    def reset(
        self,
        task_id: str,
        difficulty: Difficulty,
        data_path: str,
        seed: Optional[int] = None,
    ) -> Observation:
        """Start a new episode.

        Loads the dataset, optionally shuffles it, and returns the first
        observation.  All prior state is discarded.
        """
        self._emails = self._load_dataset(data_path)
        self._seed = seed
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self._emails)

        self._current_email_index = 0
        self._set_current_email(self._emails[0])
        self._step = 0
        self._max_steps = MAX_STEPS_PER_EPISODE
        self._done = False
        self._difficulty = difficulty
        self._task_id = task_id
        self._accumulated_reward = 0.0
        self._current_stage = Stage.CLASSIFICATION
        self._customer_sentiment = "neutral"
        self._action_history = []
        self._feedback = None

        return self._build_observation()

    # ------------------------------------------------------------------ #
    # Step helpers
    # ------------------------------------------------------------------ #

    def advance_step(self) -> None:
        """Increment step counter and check for hard episode end (loop prevention)."""
        self._step += 1
        
        # Degrade sentiment if step count grows high
        if self._step >= 2:
            self._customer_sentiment = "frustrated"
        if self._step >= 4:
            self._customer_sentiment = "angry"
            
        # Arbitrary safety limit so it doesn't run forever if baseline misbehaves
        if self._step >= 10:
            self._done = True

    def advance_stage(self) -> None:
        """Move to the next logical stage based on difficulty."""
        if self._current_stage == Stage.CLASSIFICATION:
            self._current_stage = Stage.PRIORITY
        elif self._current_stage == Stage.PRIORITY:
            if self._difficulty == Difficulty.EASY:
                self._current_stage = Stage.DONE
                self._done = True
            else:
                self._current_stage = Stage.ACTION
        elif self._current_stage == Stage.ACTION:
            if self._difficulty == Difficulty.MEDIUM:
                self._current_stage = Stage.DONE
                self._done = True
            else:
                self._current_stage = Stage.RESPONSE
        elif self._current_stage == Stage.RESPONSE:
            self._current_stage = Stage.DONE
            self._done = True

    def record_action(self, action_dict: Dict[str, Any]) -> None:
        self._action_history.append(action_dict)

    def add_reward(self, reward: float) -> None:
        self._accumulated_reward += reward

    def set_feedback(self, feedback: Optional[str]) -> None:
        self._feedback = feedback

    def mark_done(self) -> None:
        self._done = True

    # ------------------------------------------------------------------ #
    # Observation builder
    # ------------------------------------------------------------------ #

    def _set_current_email(self, email_dict: Dict[str, Any]) -> None:
        """Deep-copy email and extract ground truth to prevent leakage."""
        email_copy = copy.deepcopy(email_dict)
        self._ground_truth = {
            "classification": email_copy.pop("expected_classification", None),
            "priority": email_copy.pop("expected_priority", None),
            "action": email_copy.pop("expected_action", None),
            "response_guidelines": email_copy.pop("response_guidelines", None),
            "grading_rubric": email_copy.pop("grading_rubric", None),
        }
        self._current_email = email_copy

    def _apply_noise(self, text: str) -> str:
        """Inject typos to simulate messy real-world data."""
        if not text: return text
        words = text.split()
        for i in range(len(words)):
            if random.random() < 0.15 and len(words[i]) > 3:
                # delete a random character
                idx = random.randint(1, len(words[i]) - 2)
                words[i] = words[i][:idx] + words[i][idx+1:]
        return " ".join(words)

    def _build_observation(self) -> Observation:
        assert self._current_email is not None
        
        email_data = copy.deepcopy(self._current_email)
        
        if self._difficulty == Difficulty.HARD and self._seed is None:
            # Only inject noise if we aren't using a fixed seed to protect test stability
            email_data["body"] = self._apply_noise(email_data.get("body", ""))
            email_data["subject"] = self._apply_noise(email_data.get("subject", ""))
            
        email_obs = EmailObservation(**email_data)
        
        return Observation(
            step=self._step,
            email=email_obs,
            task_id=self._task_id,
            difficulty=self._difficulty.value,
            remaining_steps=10 - self._step,  # Arbitrary max 10 steps
            current_stage=self._current_stage.value,
            customer_sentiment=self._customer_sentiment,
            action_history=self._action_history,
            feedback=self._feedback,
        )

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    @property
    def observation(self) -> Observation:
        return self._build_observation()

    @property
    def ground_truth(self) -> Dict[str, Any]:
        assert self._ground_truth is not None
        return self._ground_truth

    @property
    def done(self) -> bool:
        return self._done

    @property
    def step_number(self) -> int:
        return self._step

    @property
    def difficulty(self) -> Difficulty:
        return self._difficulty

    @property
    def accumulated_reward(self) -> float:
        return self._accumulated_reward
        
    @property
    def current_stage(self) -> Stage:
        return self._current_stage

    @property
    def task_id(self) -> str:
        return self._task_id

    def state_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the internal state.

        This is exposed via the ``GET /state`` endpoint.  It intentionally
        omits the ground truth so agents cannot cheat.
        """
        return {
            "task_id": self._task_id,
            "difficulty": self._difficulty.value,
            "step": self._step,
            "done": self._done,
            "accumulated_reward": round(self._accumulated_reward, 4),
            "current_stage": self._current_stage.value,
            "customer_sentiment": self._customer_sentiment,
            "action_history": self._action_history,
            "current_email_id": (
                self._current_email.get("email_id") if self._current_email else None
            ),
        }
