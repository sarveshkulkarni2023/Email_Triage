"""
Rule-based grader.

Used for **easy** and **medium** tasks, and inherited by LLMGrader for base stages.
Scoring is fully deterministic for early stages.
"""

from __future__ import annotations

from typing import Any, Dict

from environment.constants import PENALTY_WRONG_ACTION
from graders.base_grader import BaseGrader
from models.action import Action


class RuleBasedGrader(BaseGrader):
    """Deterministic, rule-based grader for multi-stage MDP."""

    def grade_classification(self, action: Action, ground_truth: Dict[str, Any], weight: float) -> float:
        expected_cls = ground_truth.get("classification")
        if not expected_cls:
            return weight  # No ground truth means default to full credit
            
        cls_score = 1.0 if (
            action.classification is not None
            and action.classification.value == expected_cls
        ) else 0.0
        return cls_score * weight

    def grade_priority(self, action: Action, ground_truth: Dict[str, Any], weight: float) -> float:
        expected_pri = ground_truth.get("priority")
        if not expected_pri:
            return weight

        pri_score = 1.0 if (
            action.priority is not None
            and action.priority.value == expected_pri
        ) else 0.0

        # Partial credit: one level off → 0.5
        if pri_score == 0.0 and action.priority is not None:
            priority_order = ["critical", "high", "medium", "low"]
            try:
                diff_levels = abs(
                    priority_order.index(action.priority.value)
                    - priority_order.index(expected_pri)
                )
                if diff_levels == 1:
                    pri_score = 0.5
            except ValueError:
                pass

        return pri_score * weight

    def grade_action(self, action: Action, ground_truth: Dict[str, Any], weight: float) -> float:
        expected_act = ground_truth.get("action")
        
        if expected_act is not None:
            if action.action is not None and action.action.value == expected_act:
                return 1.0 * weight
            else:
                # Wrong action, applies penalty instead of positive reward later
                return 0.0
        else:
            # No expected action (easy tasks) — any reasonable action is fine
            return 1.0 * weight if action.action is not None else 0.0

    def grade_response(self, action: Action, ground_truth: Dict[str, Any], email_body: str, weight: float) -> float:
        # Rule-based grader does not formally grade free text responses
        # Returns 0; meant to be overridden or skipped.
        return 0.0
