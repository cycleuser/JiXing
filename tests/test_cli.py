import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from jixing.cli import (
    handle_info,
    handle_ollama_run,
    handle_search,
    handle_sessions,
    handle_stats,
    main,
    parse_args,
    setup_logging,
)


class TestSetupLogging:
    def test_default_level(self):
        setup_logging(False, False)

    def test_verbose_level(self):
        setup_logging(True, False)

    def test_quiet_level(self):
        setup_logging(False, True)

    def test_verbose_and_quiet(self):
        setup_logging(True, True)


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

    def test_sessions_list_with_provider(self):
        r = self._run_cli("sessions", "list", "--provider", "ollama")
        assert r.returncode == 0

    def test_sessions_list_with_limit(self):
        r = self._run_cli("sessions", "list", "--limit", "5")
        assert r.returncode == 0

    def test_search_command(self):
        r = self._run_cli("search", "test")
        assert r.returncode == 0

    def test_info_json_output(self):
        r = self._run_cli("--json", "info")
        assert r.returncode == 0
        assert r.stdout.strip()

    def test_stats_json_output(self):
        r = self._run_cli("--json", "stats")
        assert r.returncode == 0


class TestHandleOllamaRun:
    @patch("jixing.cli.OllamaAdapter")
    @patch("jixing.cli.SessionManager")
    def test_single_prompt(self, mock_manager, mock_adapter_class):
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.messages = []
        mock_manager.get_instance.return_value.create_session.return_value = mock_session
        mock_manager.get_instance.return_value.get_session.return_value = None

        mock_adapter = MagicMock()
        mock_adapter.run_stream.return_value = iter([
            ("Hello!", False, {}),
            ("", True, {"eval_count": 10}),
        ])
        mock_adapter_class.return_value = mock_adapter

        args = MagicMock()
        args.model = "gemma3:1b"
        args.prompt = ["Hello"]
        args.session = None
        args.interactive = False
        args.json_output = False
        args.base_url = "http://localhost:11434"
        args.compress = False

        result = handle_ollama_run(args)
        assert result == 0

    @patch("jixing.cli.OllamaAdapter")
    @patch("jixing.cli.SessionManager")
    def test_json_output(self, mock_manager, mock_adapter_class, capsys):
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.messages = []
        mock_manager.get_instance.return_value.create_session.return_value = mock_session
        mock_manager.get_instance.return_value.get_session.return_value = None

        mock_adapter = MagicMock()
        mock_adapter.run_stream.return_value = iter([
            ("Hello!", False, {}),
            ("", True, {"eval_count": 10}),
        ])
        mock_adapter_class.return_value = mock_adapter

        args = MagicMock()
        args.model = "gemma3:1b"
        args.prompt = ["Hello"]
        args.session = None
        args.interactive = False
        args.json_output = True
        args.base_url = "http://localhost:11434"
        args.compress = False

        result = handle_ollama_run(args)
        assert result == 0

    @patch("jixing.cli.OllamaAdapter")
    @patch("jixing.cli.SessionManager")
    def test_error_handling(self, mock_manager, mock_adapter_class):
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.messages = []
        mock_manager.get_instance.return_value.create_session.return_value = mock_session
        mock_manager.get_instance.return_value.get_session.return_value = None

        mock_adapter = MagicMock()
        mock_adapter.run_stream.side_effect = Exception("Connection refused")
        mock_adapter_class.return_value = mock_adapter

        args = MagicMock()
        args.model = "gemma3:1b"
        args.prompt = ["Hello"]
        args.session = None
        args.interactive = False
        args.json_output = False
        args.base_url = "http://localhost:11434"
        args.compress = False

        result = handle_ollama_run(args)
        assert result == 1

    def test_no_prompt_no_interactive(self, capsys):
        args = MagicMock()
        args.model = "gemma3:1b"
        args.prompt = []
        args.session = None
        args.interactive = False
        args.json_output = False

        result = handle_ollama_run(args)
        assert result == 1


class TestHandleSessions:
    def test_list_sessions(self, capsys):
        args = MagicMock()
        args.subcommand = "list"
        args.provider = None
        args.model = None
        args.limit = 100
        args.json_output = False

        result = handle_sessions(args)
        assert result == 0

    def test_list_sessions_with_provider(self, capsys):
        args = MagicMock()
        args.subcommand = "list"
        args.provider = "ollama"
        args.model = None
        args.limit = 100
        args.json_output = False

        result = handle_sessions(args)
        assert result == 0

    def test_get_nonexistent_session(self, capsys):
        args = MagicMock()
        args.subcommand = "get"
        args.session_id = "nonexistent"
        args.json_output = False

        result = handle_sessions(args)
        assert result == 1

    def test_delete_nonexistent_session(self, capsys):
        args = MagicMock()
        args.subcommand = "delete"
        args.session_id = "nonexistent"
        args.json_output = False

        result = handle_sessions(args)
        assert result == 1


class TestHandleStats:
    def test_stats_command(self, capsys):
        args = MagicMock()
        args.json_output = False

        result = handle_stats(args)
        assert result == 0

    def test_stats_json_output(self, capsys):
        args = MagicMock()
        args.json_output = True

        result = handle_stats(args)
        assert result == 0


class TestHandleInfo:
    def test_info_command(self, capsys):
        args = MagicMock()
        args.json_output = False

        result = handle_info(args)
        assert result == 0

    def test_info_json_output(self, capsys):
        args = MagicMock()
        args.json_output = True

        result = handle_info(args)
        assert result == 0


class TestHandleSearch:
    def test_search_command(self, capsys):
        args = MagicMock()
        args.query = "test"
        args.session = None
        args.provider = None
        args.limit = 100
        args.json_output = False

        result = handle_search(args)
        assert result == 0

    def test_search_json_output(self, capsys):
        args = MagicMock()
        args.query = "test"
        args.session = None
        args.provider = None
        args.limit = 100
        args.json_output = True

        result = handle_search(args)
        assert result == 0


class TestParseArgs:
    def test_parse_ollama_run(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "ollama", "run", "gemma3:1b", "Hello"]
            args = parse_args()
            assert args.command == "ollama"
            assert args.subcommand == "run"
            assert args.model == "gemma3:1b"
        finally:
            sys.argv = old_argv

    def test_parse_sessions_list(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "sessions", "list"]
            args = parse_args()
            assert args.command == "sessions"
            assert args.subcommand == "list"
        finally:
            sys.argv = old_argv

    def test_parse_stats(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "stats"]
            args = parse_args()
            assert args.command == "stats"
        finally:
            sys.argv = old_argv

    def test_parse_info(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "info"]
            args = parse_args()
            assert args.command == "info"
        finally:
            sys.argv = old_argv

    def test_parse_search(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "search", "query"]
            args = parse_args()
            assert args.command == "search"
            assert args.query == "query"
        finally:
            sys.argv = old_argv

    def test_parse_with_json_flag(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "--json", "stats"]
            args = parse_args()
            assert args.json_output is True
        finally:
            sys.argv = old_argv

    def test_parse_with_verbose_flag(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "-v", "stats"]
            args = parse_args()
            assert args.verbose is True
        finally:
            sys.argv = old_argv

    def test_parse_with_quiet_flag(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["jixing", "-q", "stats"]
            args = parse_args()
            assert args.quiet is True
        finally:
            sys.argv = old_argv
