"""
Observation model — what the agent sees at each step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmailObservation(BaseModel):
    """A single email presented to the agent."""

    email_id: str = Field(..., description="Unique identifier for this email.")
    sender: str = Field(..., description="Sender email address.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Full email body text.")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the email.")
    thread_history: List[str] = Field(
        default_factory=list,
        description="Prior messages in the same thread (oldest first).",
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="Filenames of attachments (metadata only).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata (e.g. customer tier, region).",
    )


class Observation(BaseModel):
    """Full observation returned by the environment."""

    step: int = Field(..., description="Current step number in the episode.")
    email: EmailObservation = Field(
        ..., description="The email to triage."
    )
    task_id: str = Field(..., description="Identifier of the current task.")
    difficulty: str = Field(..., description="easy | medium | hard")
    remaining_steps: int = Field(
        ..., description="Steps left before the episode terminates."
    )
    current_stage: str = Field(
        ..., description="Current stage of the MDP (classification, priority, action, response, done)."
    )
    customer_sentiment: str = Field(
        "neutral", description="Emotional state of the customer, degrades with errors."
    )
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of actions taken in the current episode."
    )
    feedback: Optional[str] = Field(
        None,
        description="Optional feedback from the previous step (e.g. grading hint).",
    )
