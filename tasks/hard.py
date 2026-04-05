"""
Hard task definition.

Full pipeline: classify + prioritize + act + generate response.
Includes LLM-based grading for response quality.
"""

TASK_CONFIG = {
    "id": "hard",
    "name": "Full Pipeline with Response Generation",
    "description": (
        "Complex, high-stakes emails requiring the full pipeline: "
        "classification, prioritization, action selection, and drafting "
        "a professional response. Grading is a weighted combination of "
        "rule-based scores (classification, priority, action) and "
        "LLM-evaluated response quality (relevance, correctness, completeness)."
    ),
    "data_path": "data/hard_emails.json",
    "required_fields": ["classification", "priority", "action", "response_text"],
    "optional_fields": [],
    "grading": "llm_hybrid",
    "max_steps": 5,
}
