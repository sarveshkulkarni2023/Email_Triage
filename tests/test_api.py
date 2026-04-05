"""
Tests for the FastAPI endpoints.
"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_reset_endpoint() -> None:
    response = client.post("/reset", json={"task_id": "easy", "seed": 123})
    assert response.status_code == 200
    data = response.json()
    assert "observation" in data
    
    obs = data["observation"]
    assert "email" in obs
    assert obs["task_id"] == "easy"
    assert obs["current_stage"] == "classification"


def test_step_endpoint_sequence() -> None:
    # First: Reset the environment
    client.post("/reset", json={"task_id": "medium", "seed": 42})

    # Step 1: Send Classification
    payload_1 = {"classification": "billing"}
    response_1 = client.post("/step", json=payload_1)
    
    assert response_1.status_code == 200
    data_1 = response_1.json()
    
    # Assert format
    assert "observation" in data_1
    assert "reward" in data_1
    assert "done" in data_1
    
    # Check if stage advanced assuming classification isn't penalized incorrectly
    # Even if they got it wrong, the env handles it without erroring.
    assert data_1["observation"]["current_stage"] == "priority"

    # Step 2: Send Priority
    payload_2 = {"priority": "high"}
    response_2 = client.post("/step", json=payload_2)
    assert response_2.status_code == 200
    assert response_2.json()["observation"]["current_stage"] == "action"

    # Step 3: Send Action
    payload_3 = {"action": "request_info"}
    response_3 = client.post("/step", json=payload_3)
    assert response_3.status_code == 200
    
    # Medium finishes after action
    assert response_3.json()["done"] is True


def test_state_endpoint() -> None:
    client.post("/reset", json={"task_id": "easy"})
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "accumulated_reward" in data["state"]
    # Check NO ground truth leaked
    assert "ground_truth" not in str(data)
