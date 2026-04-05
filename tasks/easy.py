"""
Easy task definition.

Clear classification + priority assignment with deterministic grading.
The agent only needs to provide ``classification`` and ``priority``.
"""

TASK_CONFIG = {
    "id": "easy",
    "name": "Clear Classification & Priority",
    "description": (
        "Straightforward emails with unambiguous intent. "
        "The agent must correctly classify the email category and assign a priority level. "
        "Scoring is fully deterministic (exact match on classification and priority)."
    ),
    "data_path": "data/easy_emails.json",
    "required_fields": ["classification", "priority"],
    "optional_fields": [],
    "grading": "rule_based",
    "max_steps": 5,
}
