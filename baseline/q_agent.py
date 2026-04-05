"""
Tabular Q-Learning Agent.

Learns a policy by discretizing the environment state (email content) 
into a small finite set of abstract features and explicitly tracking Q-values.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


# Define discrete action choices per stage
STAGE_ACTIONS = {
    "classification": ["billing", "technical_support", "account_access", "feature_request", "bug_report", "general_inquiry", "cancellation", "feedback", "security", "compliance"],
    "priority": ["critical", "high", "medium", "low"],
    "action": ["respond", "escalate", "request_info", "close", "forward"],
    "response": ["Here is the requested information.", "I have escalated this issue.", "Please provide more details.", "We are looking into this bug."]
}


class QAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0, min_epsilon: float = 0.05, decay: float = 0.995):
        self.q_table: Dict[Tuple[str, str, str], float] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def extract_state(self, observation: Dict[str, Any]) -> str:
        """Discretize the complex observation text into boolean keywords."""
        email = observation.get("email", {})
        if not email:
            return "empty_state"
            
        body = email.get("body", "").lower()
        subject = email.get("subject", "").lower()
        text = body + " " + subject
        
        # We look for simple signals to create a binary tuple string
        features = []
        features.append('1' if "password" in text or "log in" in text else '0')
        features.append('1' if "crash" in text or "bug" in text or "error" in text else '0')
        features.append('1' if "charge" in text or "billing" in text or "plan" in text else '0')
        features.append('1' if "urgent" in text or "deadline" in text or "asap" in text else '0')
        
        return "".join(features)

    def get_q_value(self, stage: str, state: str, action: str) -> float:
        return self.q_table.get((stage, state, action), 0.0)

    def get_best_action(self, stage: str, state: str) -> str:
        possible_actions = STAGE_ACTIONS.get(stage, [])
        if not possible_actions:
            return ""
            
        best_val = float("-inf")
        best_act = possible_actions[0]
        
        for a in possible_actions:
            val = self.get_q_value(stage, state, a)
            if val > best_val:
                best_val = val
                best_act = a
                
        # Tie-breaker logic (could randomly choose between maxes)
        return best_act

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Epsilon-greedy action selection."""
        stage = observation.get("current_stage", "classification")
        state = self.extract_state(observation)
        
        possible_actions = STAGE_ACTIONS.get(stage, [])
        if not possible_actions:
            return {}

        if random.random() < self.epsilon:
            # Explore
            chosen_val = random.choice(possible_actions)
        else:
            # Exploit
            chosen_val = self.get_best_action(stage, state)

        # Map to appropriate dict key
        if stage == "classification":
            return {"classification": chosen_val}
        elif stage == "priority":
            return {"priority": chosen_val}
        elif stage == "action":
            return {"action": chosen_val}
        elif stage == "response":
            return {"response_text": chosen_val}
            
        return {}

    def update(self, prev_obs: Dict[str, Any], prev_action_dict: Dict[str, Any], reward: float, next_obs: Dict[str, Any]):
        """Q-value update."""
        prev_stage = prev_obs.get("current_stage", "classification")
        prev_state = self.extract_state(prev_obs)
        
        # Get the actual string value output by the agent
        action_val = ""
        for v in prev_action_dict.values():
            if v is not None:
                action_val = v
                break
                
        if not action_val: return
        
        next_stage = next_obs.get("current_stage", "classification")
        next_state = self.extract_state(next_obs)
        
        current_q = self.get_q_value(prev_stage, prev_state, action_val)
        
        # Get max Q for next state
        if next_obs.get("done", True):
            max_next_q = 0.0
        else:
            best_next_a = self.get_best_action(next_stage, next_state)
            max_next_q = self.get_q_value(next_stage, next_state, best_next_a) if best_next_a else 0.0
            
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(prev_stage, prev_state, action_val)] = new_q
        
    def step_episode(self):
        """Decay epsilon at the end of an episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
