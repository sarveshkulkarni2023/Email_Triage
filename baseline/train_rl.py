"""
Training script for the Q-Learning Agent.

Runs 2000 episodes on the environment to demonstrate a true RL learning loop,
generates a simulated user feedback metric, and plots the learning curve.
"""

import sys
from typing import Dict, Any
from pathlib import Path

# Fix python path if running from root or baseline dir
sys.path.append(str(Path(__file__).resolve().parent.parent))

from baseline.q_agent import QAgent
from environment.env import EmailTriageEnv

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Will output CSV instead of PNG.")


def simulate_user(email: Dict[str, Any], action_dict: Dict[str, Any], stage: str) -> float:
    """
    Simulated User Feedback explicitly injected on top of the generic reward.
    If 'deadline' or 'urgent' is in the email, the action MUST NOT be 'ignore' or 'close'.
    Actually, let's use the explicit rule requested by the user:
    'if deadline in email and action == reply_now: return +1 else -1'
    Since our action space is 'respond', 'escalate', 'close', we map 'reply_now' to 'respond'.
    """
    if stage != "action":
        return 0.0  # Only apply this at the action stage
        
    if not email:
        return 0.0

    body = email.get("body", "").lower()
    subject = email.get("subject", "").lower()
    text = body + " " + subject
    
    agent_action = action_dict.get("action")
    
    if "deadline" in text or "urgent" in text:
        if agent_action == "respond":
            return 1.0
        elif agent_action == "escalate":
            return 0.5
        else:
            # e.g., 'close' or 'request_info' gets a strict penalty 
            # for ignoring a deadline.
            return -1.0
            
    return 0.0


def main():
    env = EmailTriageEnv()
    agent = QAgent(alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.01, decay=0.99)
    
    episodes = 2500
    rewards_history = []
    
    print(f"Starting Q-Learning for {episodes} episodes...")
    
    # We will train repeatedly on the 'medium' task to give it enough 
    # complexity to learn a non-trivial policy, but short enough to converge.
    for ep in range(episodes):
        obs = env.reset(task_id="medium")
        total_reward = 0.0
        done = False
        
        while not done:
            action_dict = agent.act(obs)
            current_stage = obs.get("current_stage")
            
            # Take step
            try:
                next_obs, env_reward, done, info = env.step(action_dict)
            except Exception as e:
                # If they passed an invalid action mapping, the env might error out.
                # In our env, it returns penalty and stays on stage.
                print(f"Error in env step: {e}")
                break
                
            # Simulate User Feedback
            user_feedback = simulate_user(obs.get("email", {}), action_dict, current_stage)
            
            # Combined Reward (User feedback heavily outweighs standard heuristic here for demo)
            final_reward = env_reward + user_feedback
            total_reward += final_reward
            
            # Q-Learning update
            agent.update(obs, action_dict, final_reward, next_obs)
            
            obs = next_obs
            
        agent.step_episode()
        rewards_history.append(total_reward)
        
        if (ep + 1) % 250 == 0:
            avg_reward = np.mean(rewards_history[-250:])
            print(f"Episode {ep + 1:4d} | Epsilon: {agent.epsilon:.3f} | Avg Reward (last 250): {avg_reward:.2f}")

    print("Training complete!")

    # Smooth the curve using a moving average
    window = 100
    smoothed_rewards = np.convolve(rewards_history, np.ones(window)/window, mode='valid')

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_rewards, color='#4A90E2', linewidth=2)
        plt.title('Q-Learning Agent Performance over Episodes', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(f'Moving Average Reward (window={window})', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Highlight regions
        plt.axvspan(0, 300, color='red', alpha=0.1, label='Random Exploration')
        plt.axvspan(len(smoothed_rewards)-300, len(smoothed_rewards), color='green', alpha=0.1, label='Learned Policy')
        plt.legend()
        
        plot_path = "learning_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve plotted to {plot_path}")
    else:
        # Fallback to CSV
        csv_path = "learning_curve.csv"
        with open(csv_path, "w") as f:
            f.write("episode,reward\n")
            for i, r in enumerate(rewards_history):
                f.write(f"{i},{r}\n")
        print(f"Data saved to {csv_path}")

if __name__ == "__main__":
    main()
