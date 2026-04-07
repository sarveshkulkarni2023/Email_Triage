"""
FastAPI application — the HTTP entry point for the OpenEnv environment.

Endpoints:
    POST /reset   — Reset the environment for a new episode.
    POST /step    — Submit an action and receive observation, reward, done, info.
    GET  /state   — Return the current internal state snapshot.
    GET  /health  — Health-check endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from environment.env import EmailTriageEnv

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# FastAPI app
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv-compliant environment simulating an AI customer-support "
        "email triage and response system."
    ),
    version="1.0.0",
    docs_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful per server process).
env = EmailTriageEnv()


# ------------------------------------------------------------------ #
# Request / Response schemas
# ------------------------------------------------------------------ #


class ResetRequest(BaseModel):
    task_id: str = Field(
        "easy", description="Task difficulty: 'easy', 'medium', or 'hard'."
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility."
    )


class StepRequest(BaseModel):
    classification: Optional[str] = Field(None, description="Email category.")
    priority: Optional[str] = Field(None, description="Priority level.")
    action: Optional[str] = Field(None, description="Action to take.")
    response_text: Optional[str] = Field(None, description="Drafted response.")


class ResetResponse(BaseModel):
    observation: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #


# Root endpoint is now handled implicitly by docs_url="/" in FastAPI instantiation.


@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
def reset_endpoint(request: Optional[ResetRequest] = None) -> ResetResponse:
    """Reset the environment and return the initial observation."""
    if request is None:
        request = ResetRequest()
    try:
        observation = env.reset(task_id=request.task_id, seed=request.seed)
        return ResetResponse(observation=observation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step_endpoint(request: Optional[StepRequest] = None) -> StepResponse:
    """Submit an agent action and receive the next observation."""
    if request is None:
        request = StepRequest()
    try:
        action_dict = request.model_dump(exclude_none=True)
        obs, reward, done, info = env.step(action_dict)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Step failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=StateResponse, tags=["Environment"])
def state_endpoint() -> StateResponse:
    """Return the current internal state (no ground truth leaked)."""
    return StateResponse(state=env.state())


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_endpoint() -> HealthResponse:
    """Health check."""
    return HealthResponse()


# ------------------------------------------------------------------ #
# Entrypoint (for development)
# ------------------------------------------------------------------ #

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
