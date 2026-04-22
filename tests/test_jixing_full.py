#!/usr/bin/env python3
"""Comprehensive tests for JiXing CLI and context management features.

Tests CLI commands, streaming, compression, session management, and ollama integration.

Usage:
    python tests/test_jixing_full.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))

from jixing.cli import (
    _format_size,
    handle_ollama_list,
    handle_ollama_show,
    handle_ollama_pull,
    handle_ollama_delete,
    parse_args,
)
from jixing.compressor import (
    SemanticCompressor,
    CompressionResult,
    ZH_COMPRESSION_RULES,
    EN_COMPRESSION_RULES,
)
from jixing.context_manager import (
    CompressionConfig,
    ContextWindowManager,
    MessageAnalyzer,
    RuntimeState,
    ContextSnapshot,
)
from jixing.memory import ConversationArchiver
from jixing.core import Session, SessionManager, OllamaAdapter


class TestCLIParsing(unittest.TestCase):
    """Test CLI argument parsing for all commands."""

    def test_ollama_run_basic(self):
        sys.argv = ["jixing", "ollama", "run", "gemma3:1b", "Hello"]
        args = parse_args()
        self.assertEqual(args.command, "ollama")
        self.assertEqual(args.subcommand, "run")
        self.assertEqual(args.model, "gemma3:1b")
        self.assertEqual(" ".join(args.prompt), "Hello")

    def test_ollama_run_interactive(self):
        sys.argv = ["jixing", "ollama", "run", "gemma3:1b", "-i"]
        args = parse_args()
        self.assertTrue(args.interactive)
        self.assertEqual(args.model, "gemma3:1b")

    def test_ollama_run_with_session(self):
        sys.argv = ["jixing", "ollama", "run", "gemma3:1b", "--session", "abc123", "Hi"]
        args = parse_args()
        self.assertEqual(args.session, "abc123")

    def test_ollama_run_with_compress(self):
        sys.argv = ["jixing", "ollama", "run", "gemma3:1b", "-i", "--compress"]
        args = parse_args()
        self.assertTrue(args.compress)

    def test_ollama_run_with_base_url(self):
        sys.argv = ["jixing", "ollama", "run", "gemma3:1b", "--base-url", "http://myserver:11434"]
        args = parse_args()
        self.assertEqual(args.base_url, "http://myserver:11434")

    def test_ollama_list(self):
        sys.argv = ["jixing", "ollama", "list"]
        args = parse_args()
        self.assertEqual(args.command, "ollama")
        self.assertEqual(args.subcommand, "list")

    def test_ollama_list_json(self):
        sys.argv = ["jixing", "ollama", "list", "--json"]
        args = parse_args()
        self.assertTrue(args.json_output)

    def test_ollama_show(self):
        sys.argv = ["jixing", "ollama", "show", "gemma3:1b"]
        args = parse_args()
        self.assertEqual(args.subcommand, "show")
        self.assertEqual(args.model, "gemma3:1b")

    def test_ollama_pull(self):
        sys.argv = ["jixing", "ollama", "pull", "llama3.2"]
        args = parse_args()
        self.assertEqual(args.subcommand, "pull")
        self.assertEqual(args.model, "llama3.2")

    def test_ollama_delete(self):
        sys.argv = ["jixing", "ollama", "delete", "old-model"]
        args = parse_args()
        self.assertEqual(args.subcommand, "delete")
        self.assertEqual(args.model, "old-model")

    def test_sessions_list(self):
        sys.argv = ["jixing", "sessions", "list", "--provider", "ollama"]
        args = parse_args()
        self.assertEqual(args.command, "sessions")
        self.assertEqual(args.provider, "ollama")

    def test_sessions_get(self):
        sys.argv = ["jixing", "sessions", "get", "session123"]
        args = parse_args()
        self.assertEqual(args.session_id, "session123")

    def test_sessions_delete(self):
        sys.argv = ["jixing", "sessions", "delete", "session123"]
        args = parse_args()
        self.assertEqual(args.session_id, "session123")

    def test_sessions_merge(self):
        sys.argv = ["jixing", "sessions", "merge", "s1", "s2", "--mode", "timeline"]
        args = parse_args()
        self.assertEqual(args.session_ids, ["s1", "s2"])
        self.assertEqual(args.mode, "timeline")

    def test_search(self):
        sys.argv = ["jixing", "search", "hello world"]
        args = parse_args()
        self.assertEqual(args.command, "search")
        self.assertEqual(args.query, "hello world")

    def test_stats(self):
        sys.argv = ["jixing", "stats"]
        args = parse_args()
        self.assertEqual(args.command, "stats")

    def test_info(self):
        sys.argv = ["jixing", "info"]
        args = parse_args()
        self.assertEqual(args.command, "info")

    def test_web(self):
        sys.argv = ["jixing", "web", "--port", "8080"]
        args = parse_args()
        self.assertEqual(args.command, "web")
        self.assertEqual(args.port, 8080)

    def test_version(self):
        sys.argv = ["jixing", "--version"]
        with self.assertRaises(SystemExit):
            parse_args()

    def test_json_output(self):
        sys.argv = ["jixing", "ollama", "list", "--json"]
        args = parse_args()
        self.assertTrue(args.json_output)


class TestFormatSize(unittest.TestCase):
    """Test size formatting utility."""

    def test_bytes(self):
        self.assertEqual(_format_size(500), "500 B")

    def test_kb(self):
        self.assertEqual(_format_size(1500), "1 KB")

    def test_mb(self):
        self.assertEqual(_format_size(1500000), "1 MB")

    def test_gb(self):
        self.assertEqual(_format_size(5700000000), "5.3 GB")


class TestSemanticCompressor(unittest.TestCase):
    """Test semantic text compression."""

    def setUp(self):
        self.compressor = SemanticCompressor()

    def test_compress_chinese_basic(self):
        text = "首先，人工智能具有强大的数据处理能力。其次，AI已经渗透到各行各业。综上所述，我们需要重视AI发展。"
        result = self.compressor.compress(text)
        self.assertLess(result.compressed_length, result.original_length)
        self.assertTrue(result.rules_applied)

    def test_compress_english_basic(self):
        text = "In order to achieve this goal, we need to take action. Due to the fact that AI is important, we should invest."
        result = self.compressor.compress(text)
        self.assertLess(result.compressed_length, result.original_length)

    def test_compress_empty(self):
        result = self.compressor.compress("")
        self.assertEqual(result.original_length, 0)
        self.assertEqual(result.compressed_length, 0)

    def test_compress_no_change(self):
        text = "你好世界"
        result = self.compressor.compress(text)
        self.assertEqual(result.compression_ratio, 1.0)

    def test_detect_language_chinese(self):
        text = "这是一个中文测试文本，包含很多中文字符。"
        self.assertEqual(self.compressor.detect_language(text), "zh")

    def test_detect_language_english(self):
        text = "This is an English test text with many words."
        self.assertEqual(self.compressor.detect_language(text), "en")

    def test_compress_messages(self):
        messages = [
            {"role": "user", "content": "首先，请介绍一下人工智能。" * 10},
            {"role": "assistant", "content": "综上所述，AI很重要。" * 10},
        ]
        compressed = self.compressor.compress_messages(messages)
        self.assertEqual(len(compressed), len(messages))

    def test_estimate_token_savings(self):
        text = "首先，人工智能具有强大的数据处理能力。综上所述，我们需要重视。"
        savings = self.compressor.estimate_token_savings(text)
        self.assertIn("original_tokens", savings)
        self.assertIn("compressed_tokens", savings)
        self.assertIn("token_savings", savings)

    def test_compression_result_dataclass(self):
        result = CompressionResult(
            original_text="test",
            compressed_text="tst",
            original_length=4,
            compressed_length=3,
            compression_ratio=0.75,
            rules_applied=["rule1"],
            token_reduction=1,
        )
        self.assertEqual(result.compression_ratio, 0.75)
        self.assertTrue(result.meaning_preserved)

    def test_zh_rules_loaded(self):
        self.assertGreater(len(ZH_COMPRESSION_RULES), 50)

    def test_en_rules_loaded(self):
        self.assertGreater(len(EN_COMPRESSION_RULES), 20)


class TestContextWindowManager(unittest.TestCase):
    """Test context window management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ContextWindowManager(
            context_limit=1000,
            compression_threshold=0.8,
            jsonl_dir=self.temp_dir,
            state_dir=self.temp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_messages(self, count=20, avg_length=100):
        messages = []
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Message {i}: " + "x" * avg_length
            messages.append({"role": role, "content": content, "timestamp": f"2024-01-01T00:00:{i:02d}"})
        return messages

    def test_estimate_token_count(self):
        messages = self._create_messages(10, 50)
        tokens = self.manager.estimate_token_count(messages)
        self.assertGreater(tokens, 0)

    def test_needs_compression_under_limit(self):
        messages = self._create_messages(5, 20)
        self.assertFalse(self.manager.needs_compression(messages))

    def test_needs_compression_over_limit(self):
        messages = self._create_messages(50, 100)
        self.assertTrue(self.manager.needs_compression(messages))

    def test_get_context_usage(self):
        messages = self._create_messages(10, 50)
        usage = self.manager.get_context_usage(messages)
        self.assertIn("tokens_used", usage)
        self.assertIn("context_limit", usage)
        self.assertIn("usage_percentage", usage)
        self.assertIn("needs_compression", usage)

    def test_set_compression_ratio(self):
        self.assertTrue(self.manager.set_compression_ratio(0.3))
        self.assertEqual(self.manager.config.target_ratio, 0.3)
        self.assertFalse(self.manager.set_compression_ratio(0.0))

    def test_compress_context_smart(self):
        messages = self._create_messages(20, 80)
        result = self.manager.compress_context(messages, target_ratio=0.5, goal="Implement a feature")
        self.assertTrue(result.success)
        self.assertLessEqual(result.compressed_token_count, result.original_token_count)

    def test_compress_context_preserves_goal(self):
        messages = self._create_messages(20, 80)
        goal = "Build a web application with authentication"
        result = self.manager.compress_context(messages, target_ratio=0.5, goal=goal)
        self.assertTrue(result.success)
        first_content = result.compressed_messages[0].get("content", "")
        self.assertIn(goal, first_content)

    def test_archive_and_load(self):
        messages = self._create_messages(10, 50)
        self.manager.archive_messages_to_jsonl("test_session", messages)
        loaded = self.manager.load_archived_messages("test_session")
        self.assertEqual(len(loaded), len(messages))

    def test_persist_runtime_state(self):
        path = self.manager.persist_runtime_state(
            session_id="test_session",
            goal="Build app",
            intermediate_requirements=["auth", "db"],
            decisions_made=["use postgres"],
            current_task="Implement login",
            last_round_input="Add login form",
            last_round_output="Here is the login form code",
        )
        self.assertTrue(path.exists())
        self.assertTrue(path.suffix == ".md")

    def test_load_runtime_state(self):
        path = self.manager.persist_runtime_state(
            session_id="test_session",
            goal="Build app",
            intermediate_requirements=["auth"],
            decisions_made=["use postgres"],
            current_task="Implement login",
            last_round_input="Add login",
            last_round_output="Done",
        )
        loaded = self.manager.load_runtime_state(path)
        self.assertEqual(loaded.original_goal, "Build app")

    def test_create_snapshot(self):
        messages = self._create_messages(20, 80)
        snapshot = self.manager.create_snapshot(
            "test_session", messages, goal="Build something", intermediate_requirements=["req1"]
        )
        self.assertIsNotNone(snapshot.snapshot_id)
        self.assertEqual(snapshot.full_message_count, 20)

    def test_auto_handle_overflow(self):
        messages = self._create_messages(50, 100)
        result = self.manager.auto_handle_overflow("test_session", messages, goal="test")
        self.assertIn(result["action"], [
            "compressed_light", "compressed_medium", "compressed_heavy",
            "compressed_extreme", "persisted_and_extreme_compressed"
        ])
        self.assertTrue(result["success"])


class TestConversationArchiver(unittest.TestCase):
    """Test JSONL conversation archiving."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.archiver = ConversationArchiver(archive_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_messages(self, count=10):
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}", "timestamp": f"2024-01-01T00:00:{i:02d}"}
            for i in range(count)
        ]

    def test_archive_session(self):
        messages = self._create_messages(10)
        path = self.archiver.archive_session("test_session", messages)
        self.assertTrue(path.exists())
        self.assertTrue(path.stat().st_size > 0)

    def test_load_session(self):
        messages = self._create_messages(10)
        self.archiver.archive_session("test_session", messages)
        loaded = self.archiver.load_session("test_session")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["message_count"], 10)

    def test_append_message(self):
        messages = self._create_messages(5)
        self.archiver.archive_session("test_session", messages)
        self.archiver.append_message("test_session", {"role": "user", "content": "New message"})
        loaded = self.archiver.load_session("test_session")
        self.assertEqual(loaded["message_count"], 6)

    def test_list_archived_sessions(self):
        self.archiver.archive_session("session1", self._create_messages(5))
        self.archiver.archive_session("session2", self._create_messages(10))
        sessions = self.archiver.list_archived_sessions()
        self.assertEqual(len(sessions), 2)

    def test_delete_archive(self):
        self.archiver.archive_session("test_session", self._create_messages(5))
        self.assertTrue(self.archiver.delete_archive("test_session"))
        self.assertIsNone(self.archiver.load_session("test_session"))


class TestMessageAnalyzer(unittest.TestCase):
    """Test message importance analysis."""

    def test_estimate_tokens_english(self):
        tokens = MessageAnalyzer.estimate_tokens("Hello world, this is a test.")
        self.assertGreater(tokens, 0)

    def test_estimate_tokens_chinese(self):
        tokens = MessageAnalyzer.estimate_tokens("这是一个测试消息。")
        self.assertGreater(tokens, 0)

    def test_estimate_tokens_empty(self):
        self.assertEqual(MessageAnalyzer.estimate_tokens(""), 0)

    def test_has_code_block(self):
        self.assertTrue(MessageAnalyzer.has_code_block("```python\ndef foo(): pass\n```"))
        self.assertFalse(MessageAnalyzer.has_code_block("No code here"))

    def test_compute_importance_first_user(self):
        msg = {"role": "user", "content": "Implement a sorting algorithm"}
        score = MessageAnalyzer.compute_importance(msg, 0, 10, CompressionConfig())
        self.assertGreaterEqual(score, 5.0)

    def test_compute_importance_last_message(self):
        msg = {"role": "user", "content": "Final instruction"}
        score = MessageAnalyzer.compute_importance(msg, 9, 10, CompressionConfig())
        self.assertGreaterEqual(score, 5.0)


class TestRuntimeState(unittest.TestCase):
    """Test runtime state serialization."""

    def test_to_markdown_and_back(self):
        state = RuntimeState(
            state_id="test_state",
            session_id="test_session",
            timestamp="2024-01-01T00:00:00",
            original_goal="Build app",
            intermediate_requirements=["auth", "db"],
            decisions_made=["use postgres"],
            current_task="Implement login",
            last_round_input="Add login form",
            last_round_output="Here is the code",
            pending_actions=["test"],
        )
        md = state.to_markdown()
        loaded = RuntimeState.from_markdown(md)
        self.assertEqual(loaded.original_goal, "Build app")
        self.assertEqual(loaded.current_task, "Implement login")


class TestContextSnapshot(unittest.TestCase):
    """Test context snapshot serialization."""

    def test_to_dict_and_back(self):
        snapshot = ContextSnapshot(
            snapshot_id="test_snapshot",
            source_session_id="test_session",
            timestamp="2024-01-01T00:00:00",
            original_goal="Build app",
            intermediate_requirements=["req1"],
            last_user_message="Hello",
            last_assistant_message="Hi",
            full_message_count=10,
            compressed_summary="Summary",
            compression_level=1,
        )
        data = snapshot.to_dict()
        restored = ContextSnapshot.from_dict(data)
        self.assertEqual(restored.snapshot_id, snapshot.snapshot_id)


class TestOllamaAdapter(unittest.TestCase):
    """Test Ollama adapter methods."""

    @patch("jixing.core.requests.get")
    def test_list_models(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "gemma3:1b", "size": 1000}]}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        session = Session(id="test", model_provider="ollama", model_name="test", created_at="", updated_at="", system_info={})
        adapter = OllamaAdapter(session)
        models = adapter.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "gemma3:1b")

    @patch("jixing.core.requests.post")
    def test_show_model(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"details": {"family": "llama"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        session = Session(id="test", model_provider="ollama", model_name="test", created_at="", updated_at="", system_info={})
        adapter = OllamaAdapter(session)
        info = adapter.show_model("gemma3:1b")
        self.assertIn("details", info)

    @patch("jixing.core.requests.delete")
    def test_delete_model(self, mock_delete):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_delete.return_value = mock_response

        session = Session(id="test", model_provider="ollama", model_name="test", created_at="", updated_at="", system_info={})
        adapter = OllamaAdapter(session)
        success = adapter.delete_model("old-model")
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
