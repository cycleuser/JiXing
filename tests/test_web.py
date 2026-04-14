import json
from unittest.mock import MagicMock, patch

import pytest

from jixing.web import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestWebRoutes:
    def test_index_route(self, client):
        from jixing.web import create_template_files

        create_template_files()
        response = client.get("/")
        assert response.status_code == 200

    def test_api_sessions_list(self, client):
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.get_json()
        assert "success" in data

    def test_api_sessions_list_with_filters(self, client):
        response = client.get("/api/sessions?provider=ollama&model=gemma&limit=10")
        assert response.status_code == 200
        data = response.get_json()
        assert "success" in data

    def test_api_session_get_nonexistent(self, client):
        response = client.get("/api/sessions/nonexistent")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is False

    def test_api_session_delete_nonexistent(self, client):
        response = client.delete("/api/sessions/nonexistent")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is False

    def test_api_search_no_query(self, client):
        response = client.get("/api/search")
        assert response.status_code == 400

    def test_api_search_with_query(self, client):
        response = client.get("/api/search?q=test")
        assert response.status_code == 200
        data = response.get_json()
        assert "success" in data

    def test_api_search_with_filters(self, client):
        response = client.get("/api/search?q=test&provider=ollama&limit=10")
        assert response.status_code == 200

    def test_api_stats(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert "success" in data
        assert "total_sessions" in data["data"]

    def test_api_ollama_run_no_json(self, client):
        response = client.post("/api/ollama/run")
        assert response.status_code in (400, 415)

    def test_api_ollama_run_missing_fields(self, client):
        response = client.post(
            "/api/ollama/run",
            data=json.dumps({"model": "gemma3:1b"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_api_ollama_run_missing_prompt(self, client):
        response = client.post(
            "/api/ollama/run",
            data=json.dumps({"prompt": "Hello"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_api_moxing_run_no_json(self, client):
        response = client.post("/api/moxing/run")
        assert response.status_code in (400, 415)

    def test_api_moxing_run_missing_fields(self, client):
        response = client.post(
            "/api/moxing/run",
            data=json.dumps({"model": "test-model"}),
            content_type="application/json",
        )
        assert response.status_code == 400


class TestCreateApp:
    def test_app_creation(self):
        app = create_app()
        assert app is not None
        assert app.config["JSON_AS_ASCII"] is False

    def test_app_has_routes(self):
        app = create_app()
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/" in rules
        assert "/api/sessions" in rules
        assert "/api/stats" in rules
        assert "/api/search" in rules
        assert "/api/ollama/run" in rules
        assert "/api/moxing/run" in rules
