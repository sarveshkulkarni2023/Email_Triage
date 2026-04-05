# 📧 AI Email Triage & Response System — OpenEnv Environment

An **OpenEnv-compliant** reinforcement-learning environment that simulates a real-world customer-support email inbox. Agents must classify, prioritize, decide on actions, and optionally draft professional responses for incoming emails.

---

## 📋 Table of Contents

- [Problem Description & Motivation](#-problem-description--motivation)
- [Real-World Utility](#-real-world-utility)
- [Architecture](#-architecture)
- [Observation Space](#-observation-space)
- [Action Space](#-action-space)
- [Tasks](#-tasks)
- [Reward Design](#-reward-design)
- [Setup Instructions](#-setup-instructions)
- [API Reference](#-api-reference)
- [Baseline Agent](#-baseline-agent)
- [Docker Deployment](#-docker-deployment)
- [Hugging Face Deployment](#-hugging-face-deployment)
- [Baseline Results](#-baseline-results)

---

## 🎯 Problem Description & Motivation

Customer support teams handle thousands of emails daily. Triaging them accurately — deciding the category, urgency, and appropriate action — is critical but tedious and error-prone. An AI agent that can automate this pipeline must handle:

- **Ambiguity**: Emails with unclear intent, mixed topics, or emotional tone.
- **Incomplete information**: Missing context, vague requests, unclear senders.
- **High-stakes decisions**: Compliance requests (GDPR/HIPAA), security disclosures, enterprise outages.
- **Quality communication**: Drafting responses that are empathetic, precise, and actionable.

This environment provides a structured, graded simulation for training and evaluating AI agents on these challenges.

---

## 🌍 Real-World Utility

| Application | Use Case |
|---|---|
| **Customer Support Automation** | Auto-classify and route tickets |
| **SLA Compliance** | Ensure critical emails are escalated within SLA windows |
| **Agent Training** | Train LLM agents to generate professional responses |
| **Quality Assurance** | Benchmark different triage models against a standardised rubric |
| **Process Optimisation** | Measure impact of AI triage on response time and accuracy |

---

## 🏗 Architecture

```
email-triage-env/
├── app.py                      # FastAPI entry point
├── openenv.yaml                # OpenEnv configuration
├── Dockerfile                  # Docker deployment
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
│
├── environment/
│   ├── env.py                  # Core EmailTriageEnv (reset/step/state)
│   ├── state_manager.py        # Episode lifecycle & state isolation
│   └── constants.py            # Enums, weights, penalties
│
├── models/
│   ├── observation.py          # Observation schemas (Pydantic)
│   ├── action.py               # Action schemas (Pydantic)
│   └── reward.py               # Reward breakdown schemas
│
├── graders/
│   ├── base_grader.py          # Abstract grader interface
│   ├── rule_based.py           # Deterministic grader (easy/medium)
│   └── llm_grader.py           # Hybrid grader (hard tasks)
│
├── tasks/
│   ├── easy.py                 # Task config: classification + priority
│   ├── medium.py               # Task config: + action selection
│   └── hard.py                 # Task config: + response generation
│
├── data/
│   ├── easy_emails.json        # 8 clear, unambiguous emails
│   ├── medium_emails.json      # 8 ambiguous emails
│   └── hard_emails.json        # 6 complex, high-stakes emails
│
├── baseline/
│   ├── agent.py                # OpenAI-powered baseline agent
│   └── run_baseline.py         # Runner script
│
├── utils/
│   └── helpers.py              # Shared utilities
│
└── tests/
    ├── test_environment.py     # Core env tests
    ├── test_graders.py         # Grader tests
    ├── test_state_manager.py   # State manager tests
    ├── test_models.py          # Pydantic model tests
    └── test_api.py             # API endpoint tests
```

---

## 👁 Observation Space

Each observation is a JSON object with the following structure:

```json
{
  "step": 0,
  "email": {
    "email_id": "easy-001",
    "sender": "alice@example.com",
    "subject": "Cannot access my account",
    "body": "Full email text...",
    "timestamp": "2025-03-15T09:12:00Z",
    "thread_history": ["Prior messages..."],
    "attachments": ["file.pdf"],
    "metadata": {
      "customer_tier": "enterprise",
      "region": "US-East"
    }
  },
  "task_id": "easy",
  "difficulty": "easy",
  "remaining_steps": 5,
  "feedback": null
}
```

| Field | Description |
|---|---|
| `step` | Current step number (0-indexed) |
| `email` | The email to triage (sender, subject, body, history, attachments, metadata) |
| `task_id` | Task identifier (`easy`, `medium`, `hard`) |
| `difficulty` | Difficulty level |
| `remaining_steps` | Steps left before forced termination |
| `feedback` | Feedback from the previous step's grading (or `null`) |

---

## 🎮 Action Space

The agent submits a JSON object:

```json
{
  "classification": "billing",
  "priority": "high",
  "action": "respond",
  "response_text": "Dear customer, thank you for reaching out..."
}
```

| Field | Required For | Valid Values |
|---|---|---|
| `classification` | All tasks | `billing`, `technical_support`, `account_access`, `feature_request`, `bug_report`, `general_inquiry`, `cancellation`, `feedback`, `security`, `compliance` |
| `priority` | All tasks | `critical`, `high`, `medium`, `low` |
| `action` | Medium + Hard | `respond`, `escalate`, `request_info`, `close`, `forward` |
| `response_text` | Hard only | Free-form text |

---

## 📝 Tasks

### 🟢 Easy — Clear Classification & Priority
- **Emails**: 8 straightforward emails with unambiguous intent
- **Required**: `classification` + `priority`
- **Grading**: Fully deterministic (exact match)
- **Weights**: classification 50%, priority 50%

### 🟡 Medium — Ambiguous Emails with Action Selection
- **Emails**: 8 emails with mixed signals, incomplete info, or multiple topics
- **Required**: `classification` + `priority` + `action`
- **Grading**: Rule-based with partial credit (one priority level off = 0.5)
- **Weights**: classification 30%, priority 25%, action 45%

### 🔴 Hard — Full Pipeline with Response Generation
- **Emails**: 6 complex, high-stakes scenarios (outages, GDPR, security disclosures, HIPAA)
- **Required**: `classification` + `priority` + `action` + `response_text`
- **Grading**: Rule-based (60%) + LLM-evaluated response quality (40%)
- **Weights**: classification 20%, priority 15%, action 25%, response quality 40%

---

## 💰 Reward Design

The reward function produces a score in **[0.0, 1.0]**:

```
final_reward = clamp(weighted_score - penalties, 0.0, 1.0)
```

### Component Scores
- **Classification**: 1.0 (exact match) or 0.0
- **Priority**: 1.0 (exact), 0.5 (one level off), 0.0 (otherwise)
- **Action**: 1.0 (exact match) or 0.0, with 0.15 penalty for wrong action
- **Response Quality** (hard only): Weighted average of LLM-scored relevance (35%), correctness (40%), completeness (25%)

### Penalties
| Penalty | Value | Trigger |
|---|---|---|
| Wrong action | 0.15 | Agent picks the wrong action |
| Unnecessary field | 0.05 | e.g., providing `action` on easy tasks |
| Extra step | 0.02 × step# | Each additional step beyond the first |

### Design Rationale
- **Partial rewards** for correct intermediate steps encourage incremental progress
- **Penalties for unnecessary fields** discourage over-predicting
- **Step penalties** encourage efficiency (solve in fewer steps)
- **Normalized range** enables fair comparison across tasks

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- OpenAI API key (for baseline agent and LLM grading on hard tasks)

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd email-triage-env

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run the server
python app.py
# → Server runs on http://localhost:7860

# Run tests
pytest tests/ -v
```

### Quick Test

```bash
# Reset the environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Submit an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"classification": "account_access", "priority": "high"}'

# Check state
curl http://localhost:7860/state
```

---

## 📡 API Reference

### `POST /reset`
Reset the environment for a new episode.

**Request:**
```json
{"task_id": "easy", "seed": 42}
```

**Response:**
```json
{"observation": {...}}
```

### `POST /step`
Submit an action and receive the next observation.

**Request:**
```json
{
  "classification": "billing",
  "priority": "high",
  "action": "respond",
  "response_text": "..."
}
```

**Response:**
```json
{
  "observation": {...},
  "reward": 0.85,
  "done": true,
  "info": {
    "reward_breakdown": {...},
    "accumulated_reward": 0.85
  }
}
```

### `GET /state`
Return current internal state (ground truth is never exposed).

### `GET /health`
Health check endpoint.

---

## 🤖 Baseline Agent

The baseline agent uses the OpenAI API to generate actions from observations.

```bash
# Set your API key
export OPENAI_API_KEY=sk-your-key-here

# Run baseline across all tasks
python -m baseline.run_baseline
```

The agent:
1. Receives the email observation
2. Constructs a triage prompt tailored to the difficulty level
3. Calls GPT-4o-mini with `temperature=0.0`
4. Parses the JSON response into an action dict
5. Submits to the environment

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t email-triage-env .

# Run (without LLM grading)
docker run -p 7860:7860 email-triage-env

# Run (with LLM grading for hard tasks)
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-your-key email-triage-env
```

The server will be available at `http://localhost:7860`.

---

## 🤗 Hugging Face Deployment

### Using Hugging Face Spaces (Docker)

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repository to the Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
git push hf main
```

4. Set the `OPENAI_API_KEY` secret in the Space settings (Settings → Repository secrets)
5. The app will be available at `https://YOUR_USERNAME-email-triage-env.hf.space`

### Configuration
The app is pre-configured to run on **port 7860** (Hugging Face's default).

---

## 📊 Baseline Results

Example scores from the GPT-4o-mini baseline agent:

| Task | Reward | Steps | Status |
|---|---|---|---|
| **Easy** | ~0.92 | 2 | ✅ High accuracy on clear emails |
| **Medium** | ~0.78 | 3 | ✅ Good but struggles with ambiguous cases |
| **Hard** | ~0.65 | 4 | ✅ Response quality varies |
| **Average** | ~0.78 | — | — |

> **Note**: Run `python -m baseline.run_baseline` to benchmark the LLM Agent. Exact scores depend on OpenAI's temperature and token outputs.

### Reinforcement Learning Verification (Q-Learning)
To prove empirical mathematical learning, bypass the LLM and run the tabular Q-Learning algorithm:
```bash
python baseline/train_rl.py
```
This algorithm uses $\epsilon$-greedy exploration across the sequential MDP to plot a true learning curve (improving from 0.0 to 1.0) over 2,500 episodes, saving the result to `learning_curve.png`.

> **Note**: Exact scores depend on the OpenAI model version, dataset ordering (seed), and LLM grading variability. Run `python -m baseline.run_baseline` to reproduce.

### Score Interpretation
- **0.9–1.0**: Excellent — all components correct
- **0.7–0.9**: Good — minor errors in priority or action
- **0.5–0.7**: Fair — significant errors in key components
- **0.0–0.5**: Poor — major misclassification or wrong action

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 📄 License

MIT
=======
---
title: Email Triage
emoji: 🏆
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 63adf57df18b1d82120def0f18b3317655f199ec
