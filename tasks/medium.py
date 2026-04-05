"""
Medium task definition.

Ambiguous emails requiring classification, prioritization, AND action selection.
"""

TASK_CONFIG = {
    "id": "medium",
    "name": "Ambiguous Emails with Action Selection",
    "description": (
        "Emails with ambiguous intent, mixed signals, or incomplete information. "
        "The agent must classify, prioritize, and choose an appropriate action "
        "(respond / escalate / request_info / close / forward). "
        "Grading is rule-based with partial credit for near-miss priorities."
    ),
    "data_path": "data/medium_emails.json",
    "required_fields": ["classification", "priority", "action"],
    "optional_fields": [],
    "grading": "rule_based",
    "max_steps": 5,
}
