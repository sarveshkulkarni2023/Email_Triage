"""
run_baseline.py — Run the baseline agent across all tasks and print scores.

Usage:
    python -m baseline.run_baseline

Requires OPENAI_API_KEY to be set in the environment.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List

from baseline.agent import BaselineAgent
from environment.env import EmailTriageEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TASKS = ["easy", "medium", "hard"]
SEED = 42


def run_task(
    env: EmailTriageEnv,
    agent: BaselineAgent,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Run a single task and return the results."""
    obs = env.reset(task_id=task_id, seed=seed)
    logger.info("=== Task: %s ===", task_id)

    total_reward = 0.0
    steps = 0
    done = False
    all_info: List[Dict[str, Any]] = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        all_info.append(info)
        logger.info(
            "  Step %d: reward=%.4f done=%s", steps, reward, done
        )

    return {
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "final_info": all_info[-1] if all_info else {},
    }


def main() -> None:
    env = EmailTriageEnv()
    agent = BaselineAgent()

    results: List[Dict[str, Any]] = []

    for task_id in TASKS:
        try:
            result = run_task(env, agent, task_id, seed=SEED)
            results.append(result)
        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc)
            results.append({
                "task_id": task_id,
                "error": str(exc),
                "total_reward": 0.0,
                "steps": 0,
            })

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    for r in results:
        status = "ERROR" if "error" in r else "OK"
        print(
            f"  {r['task_id']:8s} | reward={r['total_reward']:.4f} "
            f"| steps={r['steps']} | {status}"
        )

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    print(f"\n  Average reward: {avg_reward:.4f}")
    print("=" * 60)

    # Write JSON report
    report_path = "baseline/results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results written to {report_path}")


if __name__ == "__main__":
    main()
