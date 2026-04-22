#!/usr/bin/env python3
"""Comprehensive tests for context management features.

Tests context window management, compression strategies, session migration,
JSONL archiving, and runtime state persistence.

Usage:
    python tests/test_context_management.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jixing.context_manager import (
    CompressionConfig,
    CompressionResult,
    ContextSnapshot,
    ContextWindowManager,
    MessageAnalyzer,
    RuntimeState,
)
from jixing.memory import ConversationArchiver


class TestMessageAnalyzer(unittest.TestCase):
    def test_estimate_tokens_english(self):
        text = "Hello world, this is a test message with some content."
        tokens = MessageAnalyzer.estimate_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(text))

    def test_estimate_tokens_chinese(self):
        text = "这是一个测试消息，包含一些中文内容。"
        tokens = MessageAnalyzer.estimate_tokens(text)
        self.assertGreater(tokens, 0)

    def test_estimate_tokens_empty(self):
        self.assertEqual(MessageAnalyzer.estimate_tokens(""), 0)

    def test_has_code_block(self):
        self.assertTrue(MessageAnalyzer.has_code_block("```python\ndef foo():\n    pass\n```"))
        self.assertFalse(MessageAnalyzer.has_code_block("No code here"))

    def test_compute_importance_first_message(self):
        msg = {"role": "user", "content": "Implement a sorting algorithm"}
        score = MessageAnalyzer.compute_importance(msg, 0, 10, CompressionConfig())
        self.assertGreaterEqual(score, 5.0)

    def test_compute_importance_last_message(self):
        msg = {"role": "user", "content": "Final instruction"}
        score = MessageAnalyzer.compute_importance(msg, 9, 10, CompressionConfig())
        self.assertGreaterEqual(score, 5.0)

    def test_compute_importance_code_message(self):
        msg = {"role": "assistant", "content": "```python\ndef sort(arr):\n    return sorted(arr)\n```"}
        score = MessageAnalyzer.compute_importance(msg, 5, 10, CompressionConfig())
        self.assertGreaterEqual(score, 1.0)


class TestContextWindowManager(unittest.TestCase):
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
        self.assertFalse(self.manager.set_compression_ratio(1.0))

    def test_compress_context_empty(self):
        result = self.manager.compress_context([], goal="test")
        self.assertFalse(result.success)

    def test_compress_context_no_compression_needed(self):
        messages = self._create_messages(3, 20)
        result = self.manager.compress_context(messages, target_ratio=0.5, goal="test")
        self.assertTrue(result.success)

    def test_compress_context_smart(self):
        messages = self._create_messages(20, 80)
        result = self.manager.compress_context(messages, target_ratio=0.5, goal="Implement a feature")
        self.assertTrue(result.success)
        self.assertLessEqual(result.compressed_token_count, result.original_token_count)
        self.assertIn(result.strategy_used, ["smart", "preserve_critical", "progressive"])

    def test_compress_context_summary(self):
        messages = self._create_messages(20, 80)
        result = self.manager.compress_context(messages, target_ratio=0.1, goal="test")
        self.assertTrue(result.success)
        self.assertLessEqual(result.compressed_token_count, result.original_token_count)

    def test_compress_context_preserves_goal(self):
        messages = self._create_messages(20, 80)
        goal = "Build a web application with authentication"
        result = self.manager.compress_context(messages, target_ratio=0.5, goal=goal)
        self.assertTrue(result.success)
        first_content = result.compressed_messages[0].get("content", "")
        self.assertIn(goal, first_content)

    def test_compress_context_with_ai_compressor(self):
        messages = self._create_messages(20, 80)
        def fake_compressor(content):
            return f"Summary of {len(content)} characters"
        result = self.manager.compress_context(
            messages, target_ratio=0.5, goal="test", compressor_fn=fake_compressor
        )
        self.assertTrue(result.success)
        self.assertIn("Summary of", result.summary)

    def test_archive_messages_to_jsonl(self):
        messages = self._create_messages(10, 50)
        path = self.manager.archive_messages_to_jsonl("test_session", messages)
        self.assertTrue(path.exists())
        self.assertTrue(path.stat().st_size > 0)

    def test_load_archived_messages(self):
        messages = self._create_messages(10, 50)
        self.manager.archive_messages_to_jsonl("test_session", messages)
        loaded = self.manager.load_archived_messages("test_session")
        self.assertEqual(len(loaded), len(messages))

    def test_create_snapshot(self):
        messages = self._create_messages(20, 80)
        snapshot = self.manager.create_snapshot(
            "test_session", messages, goal="Build something", intermediate_requirements=["req1", "req2"]
        )
        self.assertIsNotNone(snapshot.snapshot_id)
        self.assertEqual(snapshot.source_session_id, "test_session")
        self.assertEqual(snapshot.full_message_count, 20)
        self.assertIn("Build something", snapshot.original_goal)

    def test_persist_runtime_state(self):
        path = self.manager.persist_runtime_state(
            session_id="test_session",
            goal="Build app",
            intermediate_requirements=["auth", "db"],
            decisions_made=["use postgres"],
            current_task="Implement login",
            last_round_input="Add login form",
            last_round_output="Here is the login form code",
            pending_actions=["test login"],
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
        self.assertEqual(loaded.current_task, "Implement login")

    def test_list_runtime_states(self):
        self.manager.persist_runtime_state(
            session_id="session1",
            goal="Goal 1",
            intermediate_requirements=[],
            decisions_made=[],
            current_task="Task 1",
            last_round_input="Input 1",
            last_round_output="Output 1",
        )
        self.manager.persist_runtime_state(
            session_id="session2",
            goal="Goal 2",
            intermediate_requirements=[],
            decisions_made=[],
            current_task="Task 2",
            last_round_input="Input 2",
            last_round_output="Output 2",
        )
        states = self.manager.list_runtime_states()
        self.assertGreaterEqual(len(states), 2)

    def test_build_migration_prompt(self):
        messages = self._create_messages(10, 50)
        snapshot = self.manager.create_snapshot("test_session", messages, goal="Build app")
        prompt = self.manager.build_migration_prompt(snapshot, new_goal="Continue building")
        self.assertIn("Session Migration Context", prompt)
        self.assertIn("Build app", prompt)
        self.assertIn("Continue building", prompt)

    def test_auto_handle_overflow_no_compression(self):
        messages = self._create_messages(5, 20)
        result = self.manager.auto_handle_overflow("test_session", messages, goal="test")
        self.assertEqual(result["action"], "none")
        self.assertTrue(result["success"])

    def test_auto_handle_overflow_compression(self):
        messages = self._create_messages(50, 100)
        result = self.manager.auto_handle_overflow("test_session", messages, goal="test")
        self.assertIn(result["action"], ["compressed_light", "compressed_medium", "compressed_heavy", "compressed_extreme", "persisted_and_extreme_compressed"])
        self.assertTrue(result["success"])

    def test_migration_history(self):
        messages = self._create_messages(20, 80)
        snapshot = self.manager.create_snapshot("test_session", messages, goal="test")

        class FakeSession:
            id = "new_session_123"

        migration = self.manager.migrate_session(snapshot, lambda: FakeSession())
        self.assertTrue(migration["success"])
        history = self.manager.get_migration_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["source_session_id"], "test_session")

    def test_compression_config_validation(self):
        config = CompressionConfig(target_ratio=0.5)
        self.assertTrue(config.validate())

        bad_config = CompressionConfig(target_ratio=1.5)
        self.assertFalse(bad_config.validate())

    def test_compression_hooks(self):
        hook_called = []
        def hook(result):
            hook_called.append(result)

        self.manager.register_compression_hook(hook)
        messages = self._create_messages(20, 80)
        self.manager.compress_context(messages, target_ratio=0.5, goal="test")
        self.assertEqual(len(hook_called), 1)


class TestConversationArchiver(unittest.TestCase):
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
        self.assertEqual(len(loaded["messages"]), 10)

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

    def test_load_nonexistent_session(self):
        self.assertIsNone(self.archiver.load_session("nonexistent"))


class TestRuntimeState(unittest.TestCase):
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
            context_usage={"tokens_used": 1000, "context_limit": 2000, "usage_percentage": 50},
        )
        md = state.to_markdown()
        loaded = RuntimeState.from_markdown(md)
        self.assertEqual(loaded.original_goal, "Build app")
        self.assertEqual(loaded.current_task, "Implement login")
        self.assertEqual(len(loaded.intermediate_requirements), 2)


class TestContextSnapshot(unittest.TestCase):
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
        self.assertEqual(restored.original_goal, snapshot.original_goal)


if __name__ == "__main__":
    unittest.main()
