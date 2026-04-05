"""
Tests for the EmailTriageEnv core environment.
"""

import pytest

from environment.env import EmailTriageEnv


class TestEnvironmentReset:
    """Tests for the reset() method."""

    def test_reset_easy(self) -> None:
        env = EmailTriageEnv()
        obs = env.reset(task_id="easy", seed=42)
        assert "email" in obs
        assert obs["task_id"] == "easy"
        assert obs["difficulty"] == "easy"
        assert obs["step"] == 0
        assert obs["remaining_steps"] > 0
        assert obs["current_stage"] == "classification"


class TestEnvironmentStep:
    """Tests for the step() method."""

    def test_multi_stage_sequence_easy(self) -> None:
        env = EmailTriageEnv()
        env.reset(task_id="easy", seed=42)
        
        # Note: Ground truth for easy-001 with seed=42 expects 'billing' and 'high' (hypothetically)
        # We just test the mechanics here, not the specific heuristic values.

        # Stage 1: Classification
        obs, reward, done, info = env.step({"classification": "billing"})
        assert done is False
        assert obs["current_stage"] == "priority"
        assert info["incremental_reward"] >= -0.1  # Gets some reward
        
        # Stage 2: Priority (Easy finishes after priority)
        obs, reward, done, info = env.step({"priority": "high"})
        assert done is True
        assert obs["current_stage"] == "done"

    def test_penalty_for_missing_required_stage_field(self) -> None:
        env = EmailTriageEnv()
        env.reset(task_id="medium", seed=42)

        # Stage 1: Classification
        # Sending priority but forgetting classification
        obs, reward, done, info = env.step({"priority": "high"})
        
        # Stage should NOT have advanced. Agent should be punished.
        assert done is False
        assert obs["current_stage"] == "classification"
        assert reward < 0.0  # Penalty applied
        assert "Missing classification" in info["details"]


class TestEnvironmentState:
    """Tests for the state() method."""

    def test_state_snapshot(self) -> None:
        env = EmailTriageEnv()
        env.reset(task_id="easy", seed=42)
        
        state_info = env.state()
        
        # It should contain high-level state, but NO ground truth.
        assert "task_id" in state_info
        assert "difficulty" in state_info
        assert "step" in state_info
        assert "accumulated_reward" in state_info
        assert "current_stage" in state_info
        
        # VERY IMPORTANT: Ensure ground truth is not leaked here
        assert "ground_truth" not in state_info
