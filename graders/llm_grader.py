"""
LLM-based grader.

Used for **hard** tasks.  Combines rule-based scoring for classification,
priority, and action with an LLM evaluation of the agent's drafted
response text.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional

from environment.constants import LLM_RESPONSE_WEIGHTS
from graders.rule_based import RuleBasedGrader
from models.action import Action

logger = logging.getLogger(__name__)


def _heuristic_response_score(
    response_text: str,
    guidelines: Optional[str],
) -> Dict[str, float]:
    """Fallback heuristic when no LLM is available.

    Checks keyword overlap between the response and the reference
    guidelines.  Returns relevance / correctness / completeness scores
    in ``[0, 1]``.
    """
    if not response_text:
        return {"relevance": 0.0, "correctness": 0.0, "completeness": 0.0}
    if not guidelines:
        # No guidelines to compare against → give benefit of the doubt
        return {"relevance": 0.5, "correctness": 0.5, "completeness": 0.5}

    response_words = set(response_text.lower().split())
    guideline_words = set(guidelines.lower().split())

    if not guideline_words:
        return {"relevance": 0.5, "correctness": 0.5, "completeness": 0.5}

    overlap = len(response_words & guideline_words)
    recall = overlap / len(guideline_words)
    precision = overlap / len(response_words) if response_words else 0.0

    relevance = min(1.0, precision * 2)
    completeness = min(1.0, recall)
    correctness = min(1.0, (relevance + completeness) / 2)

    return {
        "relevance": round(relevance, 4),
        "correctness": round(correctness, 4),
        "completeness": round(completeness, 4),
    }


class LLMGrader(RuleBasedGrader):
    """Hybrid grader: inherits rule-based for 1-3, LLM for stage 4."""

    def grade_response(self, action: Action, ground_truth: Dict[str, Any], email_body: str, weight: float) -> float:
        response_text = action.response_text
        if not response_text:
            return 0.0

        guidelines = ground_truth.get("response_guidelines")
        rubric = ground_truth.get("grading_rubric")
        
        scores = self._llm_evaluate_response(response_text, email_body, guidelines, rubric)
        
        raw_llm = sum(scores[criteria] * weight for criteria, weight in LLM_RESPONSE_WEIGHTS.items())
        return raw_llm * weight

    def _llm_evaluate_response(
        self,
        response_text: str,
        email_body: str,
        guidelines: Optional[str],
        rubric: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Call OpenAI to score the response on relevance, correctness,
        and completeness.  Returns a dict of floats in ``[0, 1]``.
        """
        api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("API_BASE_URL", None)
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        if not api_key:
            logger.warning("No API Key (HF_TOKEN or OPENAI_API_KEY) found — falling back to heuristic grading.")
            return _heuristic_response_score(response_text, guidelines)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            logger.error("openai package not installed. Falling back to heuristic.")
            return _heuristic_response_score(response_text, guidelines)

        prompt = (
            f"You are an expert evaluator grading a customer support agent's email response.\n\n"
            f"Original Customer Email:\n{email_body}\n\n"
            f"Reference Guidelines provided to agent:\n{guidelines or 'None'}\n\n"
            f"Specific Grading Rubric:\n{json.dumps(rubric) if rubric else 'None'}\n\n"
            f"Agent's Drafted Response:\n{response_text}\n\n"
            "Score the response on these three criteria from 0.0 (terrible) to 1.0 (perfect):\n"
            "1. relevance (Addresses the user's core issue?)\n"
            "2. correctness (Follows guidelines/rubric and states correct facts?)\n"
            "3. completeness (No missing steps or missing info?)\n\n"
            "Return EXACTLY a JSON dict: {\"relevance\": float, \"correctness\": float, \"completeness\": float}\n"
        )

        try:
            rsp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
                response_format={"type": "json_object"},
            )
            content = rsp.choices[0].message.content or "{}"
            result = json.loads(content)
            
            return {
                "relevance": float(result.get("relevance", 0.0)),
                "correctness": float(result.get("correctness", 0.0)),
                "completeness": float(result.get("completeness", 0.0)),
            }
        except Exception as exc:
            logger.exception("LLM grading failed. Falling back to heuristic.")
            return _heuristic_response_score(response_text, guidelines)
