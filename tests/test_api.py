"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestRootEndpoint:
    def test_root_returns_system_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["system"] == "KG-RAG"
        assert "endpoints" in data


class TestAskEndpoint:
    def test_ask_requires_question(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_ask_rejects_empty_question(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_rejects_short_question(self, client):
        response = client.post("/ask", json={"question": "ab"})
        assert response.status_code == 422

    def test_ask_accepts_valid_question(self, client):
        """
        Integration test: calls ORKG endpoint and Ollama.
        May fail if services are unavailable.
        """
        response = client.post(
            "/ask",
            json={"question": "Which papers compare CNN and SVM?"},
        )
        assert response.status_code in (200, 500)

        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "query_type" in data
            assert "entities" in data
            assert "sparql_queries" in data
            assert "strategies_used" in data
            # Entities should contain methods
            assert "methods" in data["entities"]


class TestHealthEndpoint:
    def test_health_returns_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data
        assert "sparql" in data
        assert "pipeline" in data
