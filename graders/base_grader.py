"""
Base grader interface.

Every grader must subclass ``BaseGrader`` and implement ``grade()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from models.action import Action
from models.reward import RewardBreakdown


class BaseGrader(ABC):
    """Abstract base class for all graders."""

    @abstractmethod
    def grade_classification(self, action: Action, ground_truth: Dict[str, Any], weight: float) -> float:
        ...

    @abstractmethod
    def grade_priority(self, action: Action, ground_truth: Dict[str, Any], weight: float) -> float:
        ...

    @abstractmethod
    def grade_action(self, action: Action, ground_truth: Dict[str, Any], weight: float) -> float:
        ...

    @abstractmethod
    def grade_response(self, action: Action, ground_truth: Dict[str, Any], email_body: str, weight: float) -> float:
        ...
