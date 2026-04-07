from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

response = client.post("/reset")
print("POST /reset with no body:")
print(response.status_code)
print(response.json())

response = client.post("/step")
print("\nPOST /step with no body:")
print(response.status_code)
print(response.json())
