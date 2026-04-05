"""
Tests for individual Grader mechanics.
"""

from typing import Any, Dict

import pytest

from graders.rule_based import RuleBasedGrader
from graders.llm_grader import LLMGrader
from models.action import Action


def test_rule_based_classification() -> None:
    grader = RuleBasedGrader()
    truth = {"classification": "billing"}
    
    # Correct action
    action = Action(classification="billing")
    score = grader.grade_classification(action, truth, weight=0.3)
    assert score == 0.3

    # Incorrect action
    action_bad = Action(classification="bug_report")
    score_bad = grader.grade_classification(action_bad, truth, weight=0.3)
    assert score_bad == 0.0


def test_rule_based_priority_partial() -> None:
    grader = RuleBasedGrader()
    truth = {"priority": "critical"}
    
    # Correct action
    action = Action(priority="critical")
    score = grader.grade_priority(action, truth, weight=0.4)
    assert score == 0.4

    # One off partial credit action
    action_partial = Action(priority="high")
    score_partial = grader.grade_priority(action_partial, truth, weight=0.4)
    assert score_partial == 0.2  # 0.5 * weight

    # Wrong action
    action_bad = Action(priority="low")
    score_bad = grader.grade_priority(action_bad, truth, weight=0.4)
    assert score_bad == 0.0


def test_rule_based_action() -> None:
    grader = RuleBasedGrader()
    truth = {"action": "escalate"}
    
    action = Action(action="escalate")
    assert grader.grade_action(action, truth, weight=1.0) == 1.0
    
    action_bad = Action(action="close")
    assert grader.grade_action(action_bad, truth, weight=1.0) == 0.0


def test_llm_grader_heuristic_fallback() -> None:
    grader = LLMGrader()
    truth = {"response_guidelines": "Apologize and mention the refund logic."}
    
    # Should use the heuristic score generator
    action = Action(response_text="I am incredibly sorry. Here is your refund.")
    score = grader.grade_response(action, truth, "My money is gone!", weight=1.0)
    
    # Will be a fuzzy heuristic number between 0 and 1
    assert 0.0 < score <= 1.0
