"""
Unit tests for FastAPI REST API endpoints.
"""

from fastapi.testclient import TestClient

from web.server import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "CV INTELLIGENCE" in response.text


def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "metrics" in data


def test_api_stream_invalid_system():
    response = client.get("/api/stream/nonexistent")
    assert response.status_code == 404
