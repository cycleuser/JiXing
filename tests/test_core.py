import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jixing.core import (
    Message,
    ModelAdapter,
    ModelRunner,
    OllamaAdapter,
    MoxingAdapter,
    Session,
    SessionManager,
    SystemInfo,
    _is_sensitive,
)


class TestIsSensitive:
    def test_api_key(self):
        assert _is_sensitive("OPENAI_API_KEY") is True

    def test_token(self):
        assert _is_sensitive("AUTH_TOKEN") is True

    def test_secret(self):
        assert _is_sensitive("APP_SECRET") is True

    def test_password(self):
        assert _is_sensitive("DB_PASSWORD") is True

    def test_credential(self):
        assert _is_sensitive("USER_CREDENTIAL") is True

    def test_auth(self):
        assert _is_sensitive("AUTH_HEADER") is True

    def test_home_not_sensitive(self):
        assert _is_sensitive("HOME") is False

    def test_path_not_sensitive(self):
        assert _is_sensitive("PATH") is False

    def test_user_not_sensitive(self):
        assert _is_sensitive("USER") is False

    def test_case_insensitive(self):
        assert _is_sensitive("openai_api_key") is True
        assert _is_sensitive("Password") is True


class TestSystemInfo:
    def test_collect_returns_system_info(self):
        info = SystemInfo.collect()
        assert isinstance(info, SystemInfo)

    def test_collect_has_hostname(self):
        info = SystemInfo.collect()
        assert info.hostname is not None
        assert len(info.hostname) > 0

    def test_collect_has_os_system(self):
        info = SystemInfo.collect()
        assert info.os_system in ("Darwin", "Linux", "Windows")

    def test_collect_has_python_version(self):
        info = SystemInfo.collect()
        assert "3." in info.python_version

    def test_collect_has_timestamp(self):
        info = SystemInfo.collect()
        assert info.timestamp is not None
        assert "T" in info.timestamp

    def test_collect_has_ip(self):
        info = SystemInfo.collect()
        assert info.network_ip is not None

    def test_collect_has_cwd(self):
        info = SystemInfo.collect()
        assert info.cwd is not None
        assert len(info.cwd) > 0

    def test_collect_filters_sensitive_env(self):
        info = SystemInfo.collect()
        for key in info.environment:
            assert not _is_sensitive(key)

    def test_to_dict(self):
        info = SystemInfo.collect()
        d = info.to_dict()
        assert isinstance(d, dict)
        assert "hostname" in d
        assert "os_system" in d
        assert "python_version" in d

    def test_environment_is_dict(self):
        info = SystemInfo.collect()
        assert isinstance(info.environment, dict)


class TestMessage:
    def test_create_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_has_timestamp(self):
        msg = Message(role="user", content="Hello")
        assert msg.timestamp is not None
        assert "T" in msg.timestamp

    def test_message_with_metrics(self):
        msg = Message(role="assistant", content="Hi", metrics={"eval_count": 50})
        assert msg.metrics == {"eval_count": 50}

    def test_message_without_metrics(self):
        msg = Message(role="user", content="Hello")
        assert msg.metrics is None

    def test_to_dict(self):
        msg = Message(role="user", content="Hello", metrics={"tokens": 10})
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert d["metrics"] == {"tokens": 10}
        assert "timestamp" in d


class TestSession:
    def test_create_session(self):
        session = Session(
            id="test-001",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        assert session.id == "test-001"
        assert session.model_provider == "ollama"
        assert session.model_name == "gemma3:1b"
        assert len(session.messages) == 0
        assert session.total_tokens == 0

    def test_add_message(self):
        session = Session(
            id="test-001",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        msg = session.add_message("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert len(session.messages) == 1

    def test_add_message_with_metrics(self):
        session = Session(
            id="test-001",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.add_message("user", "Hello", metrics={"eval_count": 20})
        session.add_message("assistant", "Hi", metrics={"eval_count": 50, "duration": 1000})
        assert session.total_tokens == 70
        assert session.total_duration_ms == 1000

    def test_add_message_updates_timestamp(self):
        session = Session(
            id="test-001",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        old_updated = session.updated_at
        time.sleep(0.01)
        session.add_message("user", "Hello")
        assert session.updated_at != old_updated

    def test_to_dict(self):
        session = Session(
            id="test-001",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={"key": "value"},
        )
        d = session.to_dict()
        assert d["id"] == "test-001"
        assert d["model_provider"] == "ollama"
        assert d["system_info"] == {"key": "value"}
        assert d["messages"] == []


class TestSessionManager:
    def test_singleton_pattern(self, tmp_db_path):
        sm1 = SessionManager.get_instance(tmp_db_path)
        sm2 = SessionManager.get_instance(tmp_db_path)
        assert sm1 is sm2

    def test_create_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        session = sm.create_session("ollama", "gemma3:1b")
        assert session.id is not None
        assert session.model_provider == "ollama"
        assert session.model_name == "gemma3:1b"

    def test_get_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        session = sm.create_session("ollama", "gemma3:1b")
        retrieved = sm.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    def test_get_nonexistent_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        assert sm.get_session("nonexistent") is None

    def test_get_current_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        assert sm.get_current_session() is None
        session = sm.create_session("ollama", "gemma3:1b")
        assert sm.get_current_session() is session

    def test_list_sessions(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sm.create_session("ollama", "gemma3:1b")
        sm.create_session("moxing", "test-model")
        sessions = sm.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_filter_by_provider(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sm.create_session("ollama", "gemma3:1b")
        sm.create_session("moxing", "test-model")
        sessions = sm.list_sessions(model_provider="ollama")
        assert len(sessions) == 1
        assert sessions[0].model_provider == "ollama"

    def test_list_sessions_filter_by_model(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sm.create_session("ollama", "gemma3:1b")
        sm.create_session("ollama", "llama2")
        sessions = sm.list_sessions(model_name="gemma")
        assert len(sessions) == 1

    def test_list_sessions_limit(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        for i in range(5):
            sm.create_session("ollama", f"model-{i}")
        sessions = sm.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_delete_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        session = sm.create_session("ollama", "gemma3:1b")
        assert sm.delete_session(session.id) is True
        assert sm.get_session(session.id) is None

    def test_delete_nonexistent_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        assert sm.delete_session("nonexistent") is False

    def test_delete_current_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        session = sm.create_session("ollama", "gemma3:1b")
        sm.delete_session(session.id)
        assert sm.get_current_session() is None

    def test_query_messages(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        session = sm.create_session("ollama", "gemma3:1b")
        session.add_message("user", "Tell me about Python")
        session.add_message("assistant", "Python is great")
        sm._save_session(session)

        results = sm.query_messages(query="Python")
        assert len(results) == 2

    def test_query_messages_by_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        session = sm.create_session("ollama", "gemma3:1b")
        session.add_message("user", "Hello")
        sm._save_session(session)

        results = sm.query_messages(session_id=session.id)
        assert len(results) == 1

    def test_query_messages_limit(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        session = sm.create_session("ollama", "gemma3:1b")
        for i in range(10):
            session.add_message("user", f"Message {i}")
        sm._save_session(session)

        results = sm.query_messages(limit=3)
        assert len(results) == 3

    def test_persistence(self, tmp_db_path):
        sm1 = SessionManager(tmp_db_path)
        session = sm1.create_session("ollama", "gemma3:1b")
        session.add_message("user", "Persistent message")
        sm1._save_session(session)

        sm2 = SessionManager(tmp_db_path)
        retrieved = sm2.get_session(session.id)
        assert retrieved is not None
        assert len(retrieved.messages) == 1


class TestModelAdapter:
    def test_abstract_parse_response(self):
        adapter = ModelAdapter(
            Session(
                id="test",
                model_provider="test",
                model_name="test",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:00:00+00:00",
                system_info={},
            )
        )
        with pytest.raises(NotImplementedError):
            adapter.parse_response(None)

    def test_prepare_messages_text_only(self):
        adapter = ModelAdapter(
            Session(
                id="test",
                model_provider="test",
                model_name="test",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:00:00+00:00",
                system_info={},
            )
        )
        messages = adapter.prepare_messages("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]


class TestOllamaAdapter:
    def test_init(self):
        session = Session(
            id="test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = OllamaAdapter(session)
        assert adapter.base_url == "http://localhost:11434"
        assert adapter.session.model_name == "gemma3:1b"

    def test_init_custom_url(self):
        session = Session(
            id="test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = OllamaAdapter(session, base_url="http://custom:11434")
        assert adapter.base_url == "http://custom:11434"

    def test_prepare_messages_text(self):
        session = Session(
            id="test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = OllamaAdapter(session)
        messages = adapter.prepare_messages("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_parse_response(self):
        session = Session(
            id="test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = OllamaAdapter(session)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Hello!",
            "eval_count": 50,
            "eval_duration": 4000000000,
            "load_duration": 500000000,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 100000000,
            "total_duration": 5000000000,
        }
        text, metrics = adapter.parse_response(mock_response)
        assert text == "Hello!"
        assert metrics["eval_count"] == 50
        assert metrics["total_duration"] == 5000000000

    @patch("requests.post")
    def test_run(self, mock_post):
        session = Session(
            id="test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = OllamaAdapter(session)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "Hello!",
            "eval_count": 50,
            "eval_duration": 4000000000,
            "load_duration": 500000000,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 100000000,
            "total_duration": 5000000000,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        text, metrics, extra = adapter.run("Hello")
        assert text == "Hello!"
        assert "wall_time_ms" in metrics
        assert extra == {}


class TestMoxingAdapter:
    def test_init(self):
        session = Session(
            id="test",
            model_provider="moxing",
            model_name="test-model",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = MoxingAdapter(session)
        assert adapter.base_url == "http://localhost:8080"

    def test_parse_response(self):
        session = Session(
            id="test",
            model_provider="moxing",
            model_name="test-model",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = MoxingAdapter(session)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {"text": "Hello from Moxing!"},
            "usage": {"completion_tokens": 30},
            "latency_ms": 250,
        }
        text, metrics = adapter.parse_response(mock_response)
        assert text == "Hello from Moxing!"
        assert metrics["tokens"] == 30
        assert metrics["latency_ms"] == 250

    @patch("requests.post")
    def test_run(self, mock_post):
        session = Session(
            id="test",
            model_provider="moxing",
            model_name="test-model",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        adapter = MoxingAdapter(session)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": {"text": "Hello!"},
            "usage": {"completion_tokens": 30},
            "latency_ms": 250,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        text, metrics, extra = adapter.run("Hello")
        assert text == "Hello!"
        assert "wall_time_ms" in metrics


class TestModelRunner:
    def test_register_adapter(self):
        class DummyAdapter(ModelAdapter):
            def parse_response(self, response):
                return "dummy", {}

            def run(self, prompt, **kwargs):
                return "dummy", {}, {}

        original = ModelRunner._adapters.copy()
        try:
            ModelRunner.register_adapter("dummy", DummyAdapter)
            assert "dummy" in ModelRunner._adapters
        finally:
            ModelRunner._adapters = original

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelRunner.run("unknown", "model", "prompt")

    @patch("jixing.core.SessionManager.get_instance")
    @patch("requests.post")
    def test_run_ollama(self, mock_post, mock_get_instance):
        mock_manager = MagicMock()
        mock_session = Session(
            id="test-session",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        mock_manager.create_session.return_value = mock_session
        mock_manager.get_instance.return_value = mock_manager
        mock_get_instance.return_value = mock_manager

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "Hello!",
            "eval_count": 50,
            "eval_duration": 4000000000,
            "load_duration": 500000000,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 100000000,
            "total_duration": 5000000000,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        session, text, metrics = ModelRunner.run("ollama", "gemma3:1b", "Hello")
        assert text == "Hello!"
        assert session.id == "test-session"
