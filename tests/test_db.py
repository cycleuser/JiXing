import json
import tempfile
from pathlib import Path

import pytest

from jixing.core import Session
from jixing.db import Database


class TestDatabaseInit:
    def test_init_creates_db_file(self, tmp_db_path):
        db = Database(tmp_db_path)
        assert tmp_db_path.exists()

    def test_init_creates_tables(self, tmp_db_path):
        db = Database(tmp_db_path)
        with db._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "sessions" in tables
            assert "messages" in tables

    def test_init_creates_indexes(self, tmp_db_path):
        db = Database(tmp_db_path)
        with db._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            assert "idx_messages_session_id" in indexes
            assert "idx_messages_timestamp" in indexes


class TestDatabaseSession:
    def test_save_session(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="test-001",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={"key": "value"},
        )
        db.save_session(session)

        with db._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", ("test-001",))
            assert cursor.fetchone()[0] == 1

    def test_save_session_with_messages(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="test-002",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        db.save_session(session)

        with db._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", ("test-002",))
            assert cursor.fetchone()[0] == 2

    def test_get_session(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="test-003",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={"os": "Linux"},
        )
        db.save_session(session)

        retrieved = db.get_session("test-003")
        assert retrieved is not None
        assert retrieved.id == "test-003"
        assert retrieved.model_provider == "ollama"
        assert retrieved.system_info == {"os": "Linux"}

    def test_get_nonexistent_session(self, tmp_db_path):
        db = Database(tmp_db_path)
        assert db.get_session("nonexistent") is None

    def test_get_session_with_messages(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="test-004",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi", metrics={"tokens": 10})
        db.save_session(session)

        retrieved = db.get_session("test-004")
        assert len(retrieved.messages) == 2
        assert retrieved.messages[0]["role"] == "user"
        assert retrieved.messages[1]["metrics"] == {"tokens": 10}

    def test_update_session(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="test-005",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        db.save_session(session)

        session.total_tokens = 100
        session.add_message("user", "New message")
        db.save_session(session)

        retrieved = db.get_session("test-005")
        assert retrieved.total_tokens == 100
        assert len(retrieved.messages) == 1


class TestDatabaseListSessions:
    def test_list_empty(self, tmp_db_path):
        db = Database(tmp_db_path)
        sessions = db.list_sessions()
        assert sessions == []

    def test_list_sessions(self, tmp_db_path):
        db = Database(tmp_db_path)
        for i in range(3):
            session = Session(
                id=f"session-{i}",
                model_provider="ollama",
                model_name="gemma3:1b",
                created_at=f"2024-01-0{i}T00:00:00+00:00",
                updated_at=f"2024-01-0{i}T00:00:00+00:00",
                system_info={},
            )
            db.save_session(session)

        sessions = db.list_sessions()
        assert len(sessions) == 3

    def test_list_filter_by_provider(self, tmp_db_path):
        db = Database(tmp_db_path)
        session1 = Session(
            id="ollama-1",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session2 = Session(
            id="moxing-1",
            model_provider="moxing",
            model_name="test-model",
            created_at="2024-01-02T00:00:00+00:00",
            updated_at="2024-01-02T00:00:00+00:00",
            system_info={},
        )
        db.save_session(session1)
        db.save_session(session2)

        ollama_sessions = db.list_sessions(model_provider="ollama")
        assert len(ollama_sessions) == 1
        assert ollama_sessions[0].model_provider == "ollama"

    def test_list_filter_by_model_name(self, tmp_db_path):
        db = Database(tmp_db_path)
        session1 = Session(
            id="gemma-session",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session2 = Session(
            id="llama-session",
            model_provider="ollama",
            model_name="llama2",
            created_at="2024-01-02T00:00:00+00:00",
            updated_at="2024-01-02T00:00:00+00:00",
            system_info={},
        )
        db.save_session(session1)
        db.save_session(session2)

        gemma_sessions = db.list_sessions(model_name="gemma")
        assert len(gemma_sessions) == 1

    def test_list_limit(self, tmp_db_path):
        db = Database(tmp_db_path)
        for i in range(10):
            session = Session(
                id=f"limit-test-{i}",
                model_provider="ollama",
                model_name="model",
                created_at=f"2024-01-{i + 1:02d}T00:00:00+00:00",
                updated_at=f"2024-01-{i + 1:02d}T00:00:00+00:00",
                system_info={},
            )
            db.save_session(session)

        sessions = db.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_ordered_by_created_at(self, tmp_db_path):
        db = Database(tmp_db_path)
        session1 = Session(
            id="older",
            model_provider="ollama",
            model_name="model",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session2 = Session(
            id="newer",
            model_provider="ollama",
            model_name="model",
            created_at="2024-01-02T00:00:00+00:00",
            updated_at="2024-01-02T00:00:00+00:00",
            system_info={},
        )
        db.save_session(session1)
        db.save_session(session2)

        sessions = db.list_sessions()
        assert sessions[0].id == "newer"


class TestDatabaseDelete:
    def test_delete_session(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="to-delete",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        db.save_session(session)

        result = db.delete_session("to-delete")
        assert result is True
        assert db.get_session("to-delete") is None

    def test_delete_nonexistent_session(self, tmp_db_path):
        db = Database(tmp_db_path)
        result = db.delete_session("nonexistent")
        assert result is False

    def test_delete_removes_messages(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="delete-with-messages",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.add_message("user", "Message 1")
        session.add_message("assistant", "Response 1")
        db.save_session(session)

        db.delete_session("delete-with-messages")

        with db._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("delete-with-messages",)
            )
            assert cursor.fetchone()[0] == 0


class TestDatabaseSearch:
    def test_search_messages(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="search-test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.add_message("user", "Tell me about Python programming")
        session.add_message("assistant", "Python is a versatile programming language")
        db.save_session(session)

        results = db.search_messages("Python")
        assert len(results) == 2

    def test_search_case_insensitive(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="case-test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.add_message("user", "PYTHON is great")
        db.save_session(session)

        results = db.search_messages("python")
        assert len(results) == 1

    def test_search_by_session(self, tmp_db_path):
        db = Database(tmp_db_path)
        session1 = Session(
            id="search-sess-1",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session1.add_message("user", "Python content")
        db.save_session(session1)

        session2 = Session(
            id="search-sess-2",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session2.add_message("user", "Python content")
        db.save_session(session2)

        results = db.search_messages("Python", session_id="search-sess-1")
        assert len(results) == 1
        assert results[0]["session_id"] == "search-sess-1"

    def test_search_by_provider(self, tmp_db_path):
        db = Database(tmp_db_path)
        session1 = Session(
            id="provider-ollama",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session1.add_message("user", "Test message")
        db.save_session(session1)

        session2 = Session(
            id="provider-moxing",
            model_provider="moxing",
            model_name="test-model",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session2.add_message("user", "Test message")
        db.save_session(session2)

        results = db.search_messages("Test", model_provider="ollama")
        assert len(results) == 1

    def test_search_limit(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="limit-search",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        for i in range(10):
            session.add_message("user", f"Test message {i}")
        db.save_session(session)

        results = db.search_messages("Test", limit=3)
        assert len(results) == 3

    def test_search_no_results(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="no-results",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.add_message("user", "Hello world")
        db.save_session(session)

        results = db.search_messages("nonexistent")
        assert results == []


class TestDatabaseStats:
    def test_stats_empty(self, tmp_db_path):
        db = Database(tmp_db_path)
        stats = db.get_stats()
        assert stats["total_sessions"] == 0
        assert stats["total_messages"] == 0
        assert stats["total_tokens"] == 0

    def test_stats_with_sessions(self, tmp_db_path):
        db = Database(tmp_db_path)
        for i in range(3):
            session = Session(
                id=f"stats-session-{i}",
                model_provider="ollama",
                model_name="gemma3:1b",
                created_at=f"2024-01-0{i}T00:00:00+00:00",
                updated_at=f"2024-01-0{i}T00:00:00+00:00",
                system_info={},
            )
            session.add_message("user", f"Message {i}")
            db.save_session(session)

        stats = db.get_stats()
        assert stats["total_sessions"] == 3
        assert stats["total_messages"] == 3

    def test_stats_token_counting(self, tmp_db_path):
        db = Database(tmp_db_path)
        session = Session(
            id="token-test",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session.total_tokens = 500
        session.total_duration_ms = 3000
        db.save_session(session)

        stats = db.get_stats()
        assert stats["total_tokens"] == 500
        assert stats["total_duration_ms"] == 3000

    def test_stats_by_provider(self, tmp_db_path):
        db = Database(tmp_db_path)
        session1 = Session(
            id="stats-ollama",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            system_info={},
        )
        session2 = Session(
            id="stats-moxing",
            model_provider="moxing",
            model_name="test-model",
            created_at="2024-01-02T00:00:00+00:00",
            updated_at="2024-01-02T00:00:00+00:00",
            system_info={},
        )
        db.save_session(session1)
        db.save_session(session2)

        stats = db.get_stats()
        assert stats["sessions_by_provider"]["ollama"] == 1
        assert stats["sessions_by_provider"]["moxing"] == 1
