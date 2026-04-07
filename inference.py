"""
Mandatory inference script for automated pre-submission evaluation.
This script utilizes the BaselineAgent against the environment and outputs strictly
formatted stdout logs ([START], [STEP], [END]) required by the hackathon organizers.
"""

import os
import json
import logging
from typing import Dict, Any

from environment.env import EmailTriageEnv
from baseline.agent import BaselineAgent

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Disable internal logging slightly so it doesn't pollute the strict stdout formats
logging.basicConfig(level=logging.ERROR)


def safe_json(data: Dict[str, Any]) -> str:
    """Helper to dump JSON strings without crashing."""
    try:
        return json.dumps(data, default=str)
    except Exception:
        return "{}"


def evaluate_task(env: EmailTriageEnv, agent: BaselineAgent, task_id: str, seed: int = 42) -> None:
    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset {task_id}: {e}")
        return

    # START format block
    start_payload = {
        "task_id": task_id,
        "difficulty": obs.get("difficulty", "unknown")
    }
    print(f"[START] {safe_json(start_payload)}")

    step = 0
    total_reward = 0.0
    done = False
    
    while not done:
        try:
            # 1. Agent acts
            action = agent.act(obs)
            
            # 2. Step the environment
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # 3. Emit exact STEP format
            step_payload = {
                "step": step,
                "action": action,
                "reward": reward,
                "stage": obs.get("current_stage", "unknown"),
                "details": info.get("details", "")
            }
            print(f"[STEP] {safe_json(step_payload)}")
            
            obs = next_obs
            step += 1
            
        except Exception as e:
            err_payload = {"step": step, "error": str(e), "reward": 0.0}
            print(f"[STEP] {safe_json(err_payload)}")
            break

    # END format block
    end_payload = {
        "task_id": task_id,
        "total_reward": total_reward,
        "steps_taken": step
    }
    print(f"[END] {safe_json(end_payload)}")


def main():
    # Enforce mandatory variable check (though we fall back gracefully)
    if not os.getenv("HF_TOKEN") and not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] Neither HF_TOKEN nor OPENAI_API_KEY set. Evaluation might fail.")

    env = EmailTriageEnv()
    agent = BaselineAgent()

    tasks = ["easy", "medium", "hard"]
    
    for t_id in tasks:
        evaluate_task(env, agent, t_id, seed=42)


if __name__ == "__main__":
    main()
