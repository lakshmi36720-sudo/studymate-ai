from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "StudyMate AI API"}


def test_ask_requires_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422
