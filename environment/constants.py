"""
Constants for the Email Triage Environment.

All enumerations, weight tables, and configuration values live here so that
no module needs to hard-code magic strings or numbers.
"""

from enum import Enum
from typing import Dict

# ---------------------------------------------------------------------------
# Email categories
# ---------------------------------------------------------------------------

class EmailCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    ACCOUNT_ACCESS = "account_access"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL_INQUIRY = "general_inquiry"
    CANCELLATION = "cancellation"
    FEEDBACK = "feedback"
    SECURITY = "security"
    COMPLIANCE = "compliance"


# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Agent actions
# ---------------------------------------------------------------------------

class AgentAction(str, Enum):
    RESPOND = "respond"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"
    CLOSE = "close"
    FORWARD = "forward"


# ---------------------------------------------------------------------------
# Task difficulty
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Episode Stages
# ---------------------------------------------------------------------------

class Stage(str, Enum):
    CLASSIFICATION = "classification"
    PRIORITY = "priority"
    ACTION = "action"
    RESPONSE = "response"
    DONE = "done"


# ---------------------------------------------------------------------------
# Episode limits
# ---------------------------------------------------------------------------

MAX_STEPS_PER_EPISODE: int = 5
DEFAULT_DIFFICULTY: Difficulty = Difficulty.EASY


# ---------------------------------------------------------------------------
# Reward weights (per difficulty)
# ---------------------------------------------------------------------------

REWARD_WEIGHTS: Dict[Difficulty, Dict[str, float]] = {
    Difficulty.EASY: {
        "classification": 0.50,
        "priority": 0.50,
        "action": 0.00,
        "response_quality": 0.00,
    },
    Difficulty.MEDIUM: {
        "classification": 0.30,
        "priority": 0.25,
        "action": 0.45,
        "response_quality": 0.00,
    },
    Difficulty.HARD: {
        "classification": 0.20,
        "priority": 0.15,
        "action": 0.25,
        "response_quality": 0.40,
    },
}


# ---------------------------------------------------------------------------
# Penalty constants
# ---------------------------------------------------------------------------

PENALTY_WRONG_ACTION: float = 0.15
PENALTY_UNNECESSARY_STEP: float = 0.05
PENALTY_PER_EXTRA_STEP: float = 0.02


# ---------------------------------------------------------------------------
# LLM grading criteria weights
# ---------------------------------------------------------------------------

LLM_RESPONSE_WEIGHTS: Dict[str, float] = {
    "relevance": 0.35,
    "correctness": 0.40,
    "completeness": 0.25,
}
