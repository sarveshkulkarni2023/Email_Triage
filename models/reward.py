"""
Reward model — structured breakdown of how the reward was computed.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class RewardBreakdown(BaseModel):
    """Detailed reward breakdown returned alongside the scalar reward."""

    classification_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Score for classification correctness."
    )
    priority_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Score for priority correctness."
    )
    action_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Score for action correctness."
    )
    response_quality_score: float = Field(
        0.0, ge=0.0, le=1.0, description="LLM-graded response quality (hard only)."
    )
    penalties: float = Field(
        0.0, ge=0.0, description="Total penalty deducted."
    )
    raw_weighted_score: float = Field(
        0.0, description="Weighted sum before penalties."
    )
    final_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Final clamped reward."
    )
    weights_used: Dict[str, float] = Field(
        default_factory=dict, description="Weights applied to each component."
    )
    details: Optional[str] = Field(
        None, description="Human-readable summary of scoring."
    )


class StepResult(BaseModel):
    """The full result of a single environment step."""

    observation: Dict = Field(..., description="Next observation (or terminal).")
    reward: float = Field(..., ge=0.0, le=1.0, description="Scalar reward.")
    done: bool = Field(..., description="Whether the episode has ended.")
    info: Dict = Field(default_factory=dict, description="Auxiliary info dict.")
