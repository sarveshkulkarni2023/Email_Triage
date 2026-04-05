"""
Tests for Pydantic models.
"""

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from models.action import Action
from models.observation import EmailObservation, Observation


def test_action_model_valid() -> None:
    # Now that fields are optional, empty actions shouldn't error
    action = Action()
    assert action.classification is None
    assert action.priority is None
    
    # Check valid values
    action = Action(classification="billing", priority="high", action="respond")
    assert action.classification.value == "billing"
    assert action.priority.value == "high"
    assert action.action.value == "respond"


def test_action_model_invalid_enum() -> None:
    with pytest.raises(ValidationError):
        Action(classification="NOT_A_VALID_CLASS")

    with pytest.raises(ValidationError):
        Action(priority="SUPER_URGENT")


def test_observation_model_valid() -> None:
    email_data: Dict[str, Any] = {
        "email_id": "123",
        "sender": "test@example.com",
        "subject": "Hello",
        "body": "World",
        "timestamp": "2025-01-01T00:00:00Z",
        "thread_history": [],
        "attachments": [],
        "metadata": {},
    }

    obs = Observation(
        step=1,
        email=EmailObservation(**email_data),
        task_id="task-1",
        difficulty="easy",
        remaining_steps=5,
        current_stage="classification",
        customer_sentiment="neutral",
        action_history=[],
        feedback="Test feedback",
    )

    assert obs.step == 1
    assert obs.email.email_id == "123"
    assert obs.difficulty == "easy"
    assert obs.current_stage == "classification"
    assert obs.customer_sentiment == "neutral"


def test_observation_missing_required() -> None:
    with pytest.raises(ValidationError):
        Observation(
            # missing email entirely
            step=1,
            task_id="task-1",
            difficulty="easy",
            remaining_steps=5,
            current_stage="classification"
        )
