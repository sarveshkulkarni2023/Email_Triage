"""
Action model — what the agent submits at each step.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from environment.constants import AgentAction, EmailCategory, Priority


class Action(BaseModel):
    """Agent action for a single step."""

    classification: Optional[EmailCategory] = Field(
        None,
        description="Predicted email category.",
    )
    priority: Optional[Priority] = Field(
        None,
        description="Predicted priority level.",
    )
    action: Optional[AgentAction] = Field(
        None,
        description="Selected action to take on the email.",
    )
    response_text: Optional[str] = Field(
        None,
        description="Drafted response text (required for 'respond' action on hard tasks).",
    )

    @field_validator("response_text")
    @classmethod
    def strip_response(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if v == "":
                return None
        return v
