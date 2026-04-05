"""
Baseline Agent — uses OpenAI to generate actions sequentially from observations.

The agent receives a stateful observation (email + current stage) and returns a
structured action dict targeting that specific stage.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaselineAgent:
    """LLM-powered baseline agent for the sequential email triage environment."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("API_BASE_URL", None)
            
            if not api_key:
                raise EnvironmentError(
                    "Either HF_TOKEN or OPENAI_API_KEY environment variable must be set."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = os.getenv("MODEL_NAME", self.model)
        return self._client

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action dict for the current stage from the observation."""
        email = observation.get("email")
        if email is None:
            # Terminal observation — nothing to do
            return {}

        current_stage = observation.get("current_stage", "classification")
        difficulty = observation.get("difficulty", "easy")
        action_history = observation.get("action_history", [])

        sys_prompt = self._build_system_prompt(current_stage)
        user_prompt = self._build_user_prompt(email, current_stage, action_history)

        client = self._get_client()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=600,
            response_format={"type": "json_object"}
        )

        raw = completion.choices[0].message.content or "{}"
        return self._parse_response(raw)

    def _build_system_prompt(self, stage: str) -> str:
        prompt = "You are an expert customer-support AI triage agent. "
        if stage == "classification":
            prompt += (
                "Classify the email. "
                "Categories: billing, technical_support, account_access, feature_request, "
                "bug_report, general_inquiry, cancellation, feedback, security, compliance.\n"
                "Return ONLY valid JSON: {\"classification\": \"<category>\"}"
            )
        elif stage == "priority":
            prompt += (
                "Assign a priority. Levels: critical, high, medium, low.\n"
                "Return ONLY valid JSON: {\"priority\": \"<level>\"}"
            )
        elif stage == "action":
            prompt += (
                "Choose an action to resolve the issue. "
                "Actions: respond, escalate, request_info, close, forward.\n"
                "Return ONLY valid JSON: {\"action\": \"<action>\"}"
            )
        elif stage == "response":
            prompt += (
                "Draft a professional, empathetic response text.\n"
                "Return ONLY valid JSON: {\"response_text\": \"<text>\"}"
            )
        return prompt

    def _build_user_prompt(self, email: Dict[str, Any], stage: str, history: list) -> str:
        parts = [
            f"**From:** {email.get('sender', 'unknown')}",
            f"**Subject:** {email.get('subject', '(no subject)')}",
            f"**Body:**\n{email.get('body', '')}\n",
        ]
        
        if history:
            parts.append("**Previous actions taken in this episode:**")
            for past_action in history:
                parts.append(f"- {json.dumps(past_action)}")
                
        parts.append(f"\nTask: Perform the {stage} stage.")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse strict JSON output from OpenAI."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s", raw[:200])
            return {}
