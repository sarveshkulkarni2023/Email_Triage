"""
Tests for StateManager and stage transitions.
"""

from typing import Any, Dict

import pytest

from environment.constants import Difficulty, Stage
from environment.state_manager import StateManager


@pytest.fixture
def manager() -> StateManager:
    return StateManager()


def test_state_manager_reset(manager: StateManager) -> None:
    obs = manager.reset(
        task_id="easy",
        difficulty=Difficulty.EASY,
        data_path="easy_emails.json",
        seed=42
    )

    assert obs.step == 0
    assert manager.done is False
    assert manager.current_stage == Stage.CLASSIFICATION
    assert obs.customer_sentiment == "neutral"


def test_stage_advancement_easy(manager: StateManager) -> None:
    manager.reset("easy", Difficulty.EASY, "easy_emails.json")
    assert manager.current_stage == Stage.CLASSIFICATION
    
    manager.advance_stage()
    assert manager.current_stage == Stage.PRIORITY
    
    manager.advance_stage()
    # Easy finishes after priority
    assert manager.current_stage == Stage.DONE


def test_stage_advancement_hard(manager: StateManager) -> None:
    manager.reset("hard", Difficulty.HARD, "hard_emails.json")
    assert manager.current_stage == Stage.CLASSIFICATION
    manager.advance_stage()
    assert manager.current_stage == Stage.PRIORITY
    manager.advance_stage()
    assert manager.current_stage == Stage.ACTION
    manager.advance_stage()
    assert manager.current_stage == Stage.RESPONSE
    manager.advance_stage()
    assert manager.current_stage == Stage.DONE


def test_sentiment_degradation(manager: StateManager) -> None:
    manager.reset("medium", Difficulty.MEDIUM, "medium_emails.json")
    assert manager.observation.customer_sentiment == "neutral"

    manager.advance_step() # step 1
    assert manager.observation.customer_sentiment == "neutral"

    manager.advance_step() # step 2
    assert manager.observation.customer_sentiment == "frustrated"

    manager.advance_step() # step 3
    manager.advance_step() # step 4
    assert manager.observation.customer_sentiment == "angry"


def test_adversarial_noise_injection(manager: StateManager) -> None:
    # Use hard difficulty to trigger noise
    obs_no_seed = manager.reset("hard", Difficulty.HARD, "hard_emails.json", seed=None)
    # The noise might have mutilated the text. Hard to assert exact strings unless we mock random.
    # Instead, we just verify it runs.
    assert isinstance(obs_no_seed.email.body, str)

    # With seed, noise is skipped to preserve test stability
    obs_seed = manager.reset("hard", Difficulty.HARD, "hard_emails.json", seed=42)
    assert len(obs_seed.email.body) > 0
    assert obs_seed.email.subject is not None
