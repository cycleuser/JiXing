import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestToolResult:
    def test_success_result(self):
        from jixing.api import ToolResult

        r = ToolResult(success=True, data={"key": "value"}, metadata={"v": "1"})
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.error is None

    def test_failure_result(self):
        from jixing.api import ToolResult

        r = ToolResult(success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"
        assert r.data is None

    def test_to_dict(self):
        from jixing.api import ToolResult

        r = ToolResult(success=True, data=[1, 2], metadata={"x": 1})
        d = r.to_dict()
        assert set(d.keys()) == {"success", "data", "error", "metadata"}

    def test_default_metadata_isolation(self):
        from jixing.api import ToolResult

        r1 = ToolResult(success=True)
        r2 = ToolResult(success=True)
        r1.metadata["a"] = 1
        assert "a" not in r2.metadata


class TestSessionManagement:
    def test_create_session(self):
        from jixing.core import Session, SystemInfo

        session = Session(
            id="test-id",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            system_info=SystemInfo.collect().to_dict(),
        )
        assert session.id == "test-id"
        assert session.model_provider == "ollama"
        assert len(session.messages) == 0

    def test_add_message(self):
        from jixing.core import Session

        session = Session(
            id="test-id",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            system_info={},
        )
        msg = session.add_message("user", "Hello", metrics={"eval_count": 10})
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert len(session.messages) == 1

    def test_session_to_dict(self):
        from jixing.core import Session

        session = Session(
            id="test-id",
            model_provider="ollama",
            model_name="gemma3:1b",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            system_info={},
        )
        d = session.to_dict()
        assert d["id"] == "test-id"
        assert d["model_provider"] == "ollama"


class TestSystemInfo:
    def test_collect(self):
        from jixing.core import SystemInfo

        info = SystemInfo.collect()
        assert info.hostname is not None
        assert info.os_system is not None
        assert info.python_version is not None
        assert info.network_ip is not None

    def test_is_sensitive(self):
        from jixing.core import _is_sensitive

        assert _is_sensitive("OPENAI_API_KEY") is True
        assert _is_sensitive("PASSWORD") is True
        assert _is_sensitive("HOME") is False


class TestAPI:
    @patch("jixing.core.SessionManager.get_instance")
    def test_run_ollama_success(self, mock_get_instance):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.messages = []
        mock_session.to_dict.return_value = {"id": "test-session"}
        mock_manager.create_session.return_value = mock_session
        mock_manager.get_session.return_value = mock_session
        mock_get_instance.return_value = mock_manager

        with patch("jixing.core.ModelRunner.run") as mock_run:
            mock_run.return_value = (mock_session, "Hello!", {"eval_count": 10})
            from jixing.api import run_ollama

            result = run_ollama(model="gemma3:1b", prompt="Hi")
            assert result.success is True

    def test_query_sessions(self):
        from jixing.api import query_sessions

        result = query_sessions(limit=10)
        assert result.success is True
        assert "data" in result.to_dict()

    def test_get_system_info(self):
        from jixing.api import get_system_info

        result = get_system_info()
        assert result.success is True
        assert result.data is not None
        assert "hostname" in result.data


class TestDatabase:
    def test_db_init_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from jixing.core import Session
            from jixing.db import Database

            db = Database(Path(tmpdir) / "test.db")
            session = Session(
                id="db-test-id",
                model_provider="ollama",
                model_name="test-model",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                system_info={},
            )
            session.add_message("user", "test message")

            db.save_session(session)
            loaded = db.get_session("db-test-id")

            assert loaded is not None
            assert loaded.id == "db-test-id"
            assert len(loaded.messages) == 1

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from jixing.core import Session
            from jixing.db import Database

            db = Database(Path(tmpdir) / "test.db")

            for i in range(3):
                session = Session(
                    id=f"session-{i}",
                    model_provider="ollama",
                    model_name="test-model",
                    created_at="2024-01-01T00:00:00Z",
                    updated_at="2024-01-01T00:00:00Z",
                    system_info={},
                )
                db.save_session(session)

            sessions = db.list_sessions()
            assert len(sessions) == 3

    def test_search_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from jixing.core import Session
            from jixing.db import Database

            db = Database(Path(tmpdir) / "test.db")
            session = Session(
                id="search-test",
                model_provider="ollama",
                model_name="test-model",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                system_info={},
            )
            session.add_message("user", "Tell me about Python programming")
            session.add_message("assistant", "Python is a great language.")
            db.save_session(session)

            results = db.search_messages("Python")
            assert len(results) == 2


class TestToolsSchema:
    def test_tools_is_list(self):
        from jixing.tools import TOOLS

        assert isinstance(TOOLS, list)
        assert len(TOOLS) >= 1

    def test_tool_names(self):
        from jixing.tools import TOOLS

        for tool in TOOLS:
            name = tool["function"]["name"]
            assert name.startswith("jixing_")

    def test_tool_structure(self):
        from jixing.tools import TOOLS

        for tool in TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "properties" in func["parameters"]
            assert "required" in func["parameters"]

    def test_required_fields_in_properties(self):
        from jixing.tools import TOOLS

        for tool in TOOLS:
            func = tool["function"]
            props = func["parameters"]["properties"]
            for req in func["parameters"]["required"]:
                assert req in props, f"Required '{req}' not in properties"


class TestToolsDispatch:
    def test_dispatch_unknown_tool(self):
        from jixing.tools import dispatch

        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("nonexistent_tool", {})

    def test_dispatch_json_string_args(self):
        from jixing.tools import dispatch

        args = json.dumps({"query": "test"})
        result = dispatch("jixing_search_messages", args)
        assert isinstance(result, dict)
        assert "success" in result

    def test_dispatch_dict_args(self):
        from jixing.tools import dispatch

        result = dispatch("jixing_get_stats", {})
        assert "success" in result

    def test_dispatch_run_ollama_args(self):
        from jixing.tools import dispatch

        args = json.dumps({"model": "gemma3:1b", "prompt": "Hello"})
        with patch("jixing.api.run_ollama") as mock_run:
            mock_run.return_value = MagicMock(success=True, to_dict=lambda: {"success": True})
            result = dispatch("jixing_run_ollama", args)
            assert "success" in result


class TestCLIFlags:
    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "jixing"] + list(args),
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_version_flag(self):
        r = self._run_cli("-V")
        assert r.returncode == 0
        assert "jixing" in r.stdout.lower() or "jixing" in r.stderr.lower()

    def test_help_has_unified_flags(self):
        r = self._run_cli("--help")
        assert r.returncode == 0
        assert "--json" in r.stdout
        assert "--quiet" in r.stdout or "-q" in r.stdout
        assert "--verbose" in r.stdout or "-v" in r.stdout

    def test_info_command(self):
        r = self._run_cli("info")
        assert r.returncode == 0

    def test_stats_command(self):
        r = self._run_cli("stats")
        assert r.returncode == 0

    def test_sessions_list_command(self):
        r = self._run_cli("sessions", "list")
        assert r.returncode == 0


class TestPackageExports:
    def test_version(self):
        import jixing

        assert hasattr(jixing, "__version__")
        assert isinstance(jixing.__version__, str)

    def test_toolresult(self):
        from jixing import ToolResult

        assert callable(ToolResult)

    def test_all_defined(self):
        import jixing

        assert hasattr(jixing, "__all__")


class TestModels:
    def test_ollama_adapter_init(self):
        from jixing.models.ollama import OllamaModel

        model = OllamaModel("gemma3:1b")
        assert model.model_name == "gemma3:1b"
        assert "http://localhost:11434" in model.base_url

    def test_moxing_adapter_init(self):
        from jixing.models.moxing import MoxingModel

        model = MoxingModel("test-model")
        assert model.model_name == "test-model"
        assert "http://localhost:8080" in model.base_url

    def test_base_adapter(self):
        from jixing.models.base import ModelAdapterBase

        class DummyAdapter(ModelAdapterBase):
            def chat(self, messages, **kwargs):
                return "", {}

            def generate(self, prompt, **kwargs):
                return "", {}

        adapter = DummyAdapter("test")
        assert adapter.model_name == "test"
        config = adapter.get_config()
        assert config == {}
