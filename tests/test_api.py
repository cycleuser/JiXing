import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jixing.api import (
    ToolResult,
    delete_session,
    get_session,
    get_stats,
    get_system_info,
    query_sessions,
    run_moxing,
    run_ollama,
    search_messages,
)


class TestToolResult:
    def test_success_result(self):
        r = ToolResult(success=True, data={"key": "value"}, metadata={"v": "1"})
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.error is None

    def test_failure_result(self):
        r = ToolResult(success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"
        assert r.data is None

    def test_to_dict(self):
        r = ToolResult(success=True, data=[1, 2], metadata={"x": 1})
        d = r.to_dict()
        assert set(d.keys()) == {"success", "data", "error", "metadata"}

    def test_default_metadata_isolation(self):
        r1 = ToolResult(success=True)
        r2 = ToolResult(success=True)
        r1.metadata["a"] = 1
        assert "a" not in r2.metadata

    def test_to_dict_success(self):
        r = ToolResult(success=True, data={"result": "ok"})
        d = r.to_dict()
        assert d["success"] is True
        assert d["data"] == {"result": "ok"}

    def test_to_dict_error(self):
        r = ToolResult(success=False, error="failed")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "failed"

    def test_empty_result(self):
        r = ToolResult(success=True)
        d = r.to_dict()
        assert d["success"] is True
        assert d["data"] is None
        assert d["error"] is None


class TestRunOllama:
    @patch("jixing.core.SessionManager.get_instance")
    @patch("jixing.core.ModelRunner.run")
    def test_success(self, mock_run, mock_get_instance):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.messages = []
        mock_session.to_dict.return_value = {"id": "test-session"}
        mock_manager.create_session.return_value = mock_session
        mock_get_instance.return_value = mock_manager

        mock_run.return_value = (mock_session, "Hello!", {"eval_count": 10})

        result = run_ollama(model="gemma3:1b", prompt="Hi")
        assert result.success is True
        assert result.data["response"] == "Hello!"

    @patch("jixing.core.SessionManager.get_instance")
    @patch("jixing.core.ModelRunner.run")
    def test_with_existing_session(self, mock_run, mock_get_instance):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.id = "existing-session"
        mock_session.messages = []
        mock_manager.get_session.return_value = mock_session
        mock_get_instance.return_value = mock_manager

        mock_run.return_value = (mock_session, "Response", {})

        result = run_ollama(model="gemma3:1b", prompt="Hi", session_id="existing-session")
        assert result.success is True

    @patch("jixing.core.SessionManager.get_instance")
    @patch("jixing.core.ModelRunner.run")
    def test_exception_handling(self, mock_run, mock_get_instance):
        mock_get_instance.side_effect = Exception("DB error")

        result = run_ollama(model="gemma3:1b", prompt="Hi")
        assert result.success is False
        assert "DB error" in result.error


class TestRunMoxing:
    @patch("jixing.core.SessionManager.get_instance")
    @patch("jixing.core.ModelRunner.run")
    def test_success(self, mock_run, mock_get_instance):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.messages = []
        mock_manager.create_session.return_value = mock_session
        mock_get_instance.return_value = mock_manager

        mock_run.return_value = (mock_session, "Moxing response!", {"tokens": 20})

        result = run_moxing(model="test-model", prompt="Hi")
        assert result.success is True
        assert result.data["response"] == "Moxing response!"

    @patch("jixing.core.SessionManager.get_instance")
    @patch("jixing.core.ModelRunner.run")
    def test_exception_handling(self, mock_run, mock_get_instance):
        mock_get_instance.side_effect = Exception("Connection error")

        result = run_moxing(model="test-model", prompt="Hi")
        assert result.success is False


class TestQuerySessions:
    def test_query_sessions(self):
        result = query_sessions(limit=10)
        assert result.success is True
        assert isinstance(result.data, list)

    def test_query_with_provider_filter(self):
        result = query_sessions(model_provider="ollama", limit=10)
        assert result.success is True

    def test_query_with_model_filter(self):
        result = query_sessions(model_name="gemma", limit=10)
        assert result.success is True


class TestGetSession:
    def test_get_nonexistent_session(self):
        result = get_session(session_id="nonexistent-session")
        assert result.success is False
        assert "not found" in result.error.lower()


class TestDeleteSession:
    def test_delete_nonexistent_session(self):
        result = delete_session(session_id="nonexistent-session")
        assert result.success is False


class TestSearchMessages:
    def test_search_messages(self):
        result = search_messages(query="test")
        assert result.success is True
        assert isinstance(result.data, list)

    def test_search_with_session_filter(self):
        result = search_messages(query="test", session_id="nonexistent")
        assert result.success is True

    def test_search_with_provider_filter(self):
        result = search_messages(query="test", model_provider="ollama")
        assert result.success is True


class TestGetStats:
    def test_get_stats(self):
        result = get_stats()
        assert result.success is True
        assert "total_sessions" in result.data
        assert "total_messages" in result.data


class TestGetSystemInfo:
    def test_get_system_info(self):
        result = get_system_info()
        assert result.success is True
        assert result.data is not None
        assert "hostname" in result.data
        assert "os_system" in result.data
        assert "python_version" in result.data
