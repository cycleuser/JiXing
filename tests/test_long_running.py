"""Tests for long-running task executor."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jixing.long_running_executor import (
    LongRunningTaskExecutor,
    TaskProgress,
    TaskResult,
    format_duration,
    parse_duration,
)


class TestParseDuration:
    def test_integer_seconds(self):
        assert parse_duration(300) == 300.0

    def test_float_seconds(self):
        assert parse_duration(3600.5) == 3600.5

    def test_string_seconds(self):
        assert parse_duration("300s") == 300.0
        assert parse_duration("60sec") == 60.0
        assert parse_duration("10seconds") == 10.0

    def test_minutes(self):
        assert parse_duration("5m") == 300.0
        assert parse_duration("2min") == 120.0
        assert parse_duration("10minutes") == 600.0

    def test_hours(self):
        assert parse_duration("2h") == 7200.0
        assert parse_duration("1hr") == 3600.0
        assert parse_duration("3hours") == 10800.0

    def test_days(self):
        assert parse_duration("1d") == 86400.0
        assert parse_duration("2days") == 172800.0

    def test_weeks(self):
        assert parse_duration("1w") == 604800.0
        assert parse_duration("2weeks") == 1209600.0

    def test_months(self):
        assert parse_duration("1M") == 2592000.0
        assert parse_duration("2months") == 5184000.0

    def test_years(self):
        assert parse_duration("1y") == 31536000.0
        assert parse_duration("2years") == 63072000.0

    def test_combined_duration(self):
        assert parse_duration("2h 30m") == 9000.0
        assert parse_duration("1d 12h") == 129600.0
        assert parse_duration("1w 2d") == 777600.0

    def test_unlimited(self):
        assert parse_duration("unlimited") is None
        assert parse_duration("forever") is None
        assert parse_duration("infinite") is None
        assert parse_duration(None) is None
        assert parse_duration("0") is None

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            parse_duration("invalid")

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            parse_duration([1, 2, 3])


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(30) == "30s"

    def test_minutes(self):
        assert format_duration(300) == "5.0m"

    def test_hours(self):
        assert format_duration(7200) == "2.0h"

    def test_days(self):
        assert format_duration(86400) == "1.0d"

    def test_months(self):
        assert format_duration(2592000) == "1.0M"

    def test_years(self):
        assert format_duration(31536000) == "1.0y"

    def test_none(self):
        assert format_duration(None) == "unlimited"


class TestTaskProgress:
    def test_to_dict(self):
        progress = TaskProgress(
            task_id="test123",
            goal="test goal",
            model_name="test-model",
            started_at="2024-01-01T00:00:00",
        )
        d = progress.to_dict()
        assert d["task_id"] == "test123"
        assert d["goal"] == "test goal"
        assert d["model_name"] == "test-model"
        assert d["status"] == "running"


class TestTaskResult:
    def test_to_dict(self):
        result = TaskResult(
            success=True,
            task_id="test123",
            goal="test goal",
            final_output="done",
            rounds_completed=5,
            total_tokens_used=1000,
            elapsed_seconds=60.0,
            compressions_performed=2,
            migrations_performed=1,
            session_ids=["sess1"],
            quality_score=0.9,
            stop_reason="Quality threshold reached",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["rounds_completed"] == 5
        assert d["quality_score"] == 0.9


class TestLongRunningTaskExecutor:
    @pytest.fixture
    def mock_session_manager(self):
        with patch("jixing.long_running_executor.SessionManager") as mock:
            manager = MagicMock()
            session = MagicMock()
            session.id = "test-session-id"
            session.model_name = "test-model"
            session.goal = "test goal"
            session.context_limit = 128000
            session.messages = []
            manager.create_session.return_value = session
            manager.get_instance.return_value = manager
            mock.get_instance.return_value = manager
            yield mock, manager, session

    @pytest.fixture
    def mock_ollama_adapter(self):
        with patch("jixing.long_running_executor.OllamaAdapter") as mock:
            adapter = MagicMock()
            adapter.run.return_value = ("test response", {"eval_count": 100}, {})
            mock.return_value = adapter
            yield mock, adapter

    def test_init_creates_session(self, mock_session_manager, mock_ollama_adapter):
        _, manager, session = mock_session_manager
        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
        )
        manager.create_session.assert_called_once()
        assert executor.task_id is not None
        assert executor.model_name == "test-model"
        assert executor.goal == "test goal"

    def test_execute_single_round(self, mock_session_manager, mock_ollama_adapter):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        adapter.run.return_value = ("[COMPLETE] Task done", {"eval_count": 50}, {})

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=5,
        )

        result = executor.execute()

        assert result.success is True
        assert result.rounds_completed == 1
        assert "[COMPLETE]" in result.final_output

    def test_execute_stops_on_quality(self, mock_session_manager, mock_ollama_adapter):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        adapter.run.return_value = ("Good response", {"eval_count": 50}, {})

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            quality_threshold=0.0,
            max_rounds=10,
        )

        result = executor.execute()

        assert result.rounds_completed >= 1

    def test_execute_stops_on_time_limit(self, mock_session_manager, mock_ollama_adapter):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        adapter.run.return_value = ("Working...", {"eval_count": 50}, {})

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_duration="0.1s",
            max_rounds=100,
        )

        executor.start_time = time.time() - 0.2

        result = executor.execute()

        assert "Time limit" in result.stop_reason

    def test_execute_stops_on_round_limit(self, mock_session_manager, mock_ollama_adapter):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        adapter.run.return_value = ("Working...", {"eval_count": 50}, {})

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=3,
            quality_threshold=1.0,
        )

        result = executor.execute()

        assert result.rounds_completed == 3
        assert "Round limit" in result.stop_reason

    def test_stop_method(self, mock_session_manager, mock_ollama_adapter):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        adapter.run.return_value = ("Working...", {"eval_count": 50}, {})

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=100,
        )

        executor.stop()
        assert executor._running is False

    def test_progress_callback(self, mock_session_manager, mock_ollama_adapter):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        adapter.run.return_value = ("[COMPLETE] done", {"eval_count": 50}, {})

        progress_calls = []

        def callback(progress):
            progress_calls.append(progress)

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
            progress_callback=callback,
        )

        executor.execute()

        assert len(progress_calls) >= 1

    def test_checkpoint_saved(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        adapter.run.return_value = ("[COMPLETE] done", {"eval_count": 50}, {})

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
            checkpoint_dir=str(tmp_path),
        )

        executor.execute()

        checkpoint_files = list(tmp_path.glob("*.json"))
        assert len(checkpoint_files) >= 1

    def test_from_checkpoint(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        checkpoint = {
            "task_id": "test-task",
            "round": 2,
            "session_id": "test-session-id",
            "total_tokens": 200,
        }
        cp_path = tmp_path / "checkpoint.json"
        with open(cp_path, "w") as f:
            json.dump(checkpoint, f)

        manager.get_session.return_value = session

        executor = LongRunningTaskExecutor.from_checkpoint(
            cp_path,
            max_rounds=1,
        )

        assert executor.task_id == "test-task"
        assert executor.rounds_completed == 2
        assert executor.total_tokens_used == 200

    def test_from_checkpoint_missing_file(self):
        with pytest.raises(FileNotFoundError):
            LongRunningTaskExecutor.from_checkpoint("/nonexistent/path.json")


class TestFileExtraction:
    """Test code extraction and file writing logic."""

    @pytest.fixture
    def mock_session_manager(self):
        with patch("jixing.long_running_executor.SessionManager") as mock:
            manager = MagicMock()
            session = MagicMock()
            session.id = "test-session-id"
            session.model_name = "test-model"
            session.goal = "test goal"
            session.context_limit = 128000
            session.messages = []
            manager.create_session.return_value = session
            manager.get_instance.return_value = manager
            mock.get_instance.return_value = manager
            yield mock, manager, session

    @pytest.fixture
    def mock_ollama_adapter(self):
        with patch("jixing.long_running_executor.OllamaAdapter") as mock:
            adapter = MagicMock()
            adapter.run.return_value = ("test response", {"eval_count": 100}, {})
            adapter.show_model.return_value = {"model_info": {"general.parameter_count": "1B"}}
            mock.return_value = adapter
            yield mock, adapter

    def test_extract_python_code_block(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test extracting code from ```python block."""
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
            work_dir=tmp_path,
        )

        text = """Here's the code:
```python
import pygame
pygame.init()
while running:
    pass
```
"""
        files = executor._extract_and_write_files(text)
        assert len(files) == 1
        assert "tank.py" in files[0] or "main.py" in files[0]

    def test_extract_code_with_file_path(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test extracting code from ```path/to/file.py block."""
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
            work_dir=tmp_path,
        )

        text = """Complete code:
```tank_game/main.py
import pygame
pygame.init()
if __name__ == "__main__":
    main()
```
"""
        files = executor._extract_and_write_files(text)
        assert len(files) == 1
        assert "tank_game/main.py" in files[0]
        assert (tmp_path / "tank_game" / "main.py").exists()

    def test_extract_multiple_files(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test extracting multiple code blocks."""
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
            work_dir=tmp_path,
        )

        text = """Main file:
```game/main.py
import pygame
pygame.init()
```

Readme:
```game/README.md
# Game
```
"""
        files = executor._extract_and_write_files(text)
        assert len(files) == 2

    def test_skip_bash_blocks(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test that bash blocks are not written as files."""
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
            work_dir=tmp_path,
        )

        text = """Install command:
```bash
pip install pygame
```

Code:
```main.py
import pygame
```
"""
        files = executor._extract_and_write_files(text)
        assert len(files) == 1
        assert "main.py" in files[0]

    def test_is_main_script(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test main script detection."""
        _, manager, session = mock_session_manager
        _, adapter = mock_ollama_adapter

        executor = LongRunningTaskExecutor(
            model_name="test-model",
            goal="test goal",
            max_rounds=1,
            work_dir=tmp_path,
        )

        # Should be main script (has pygame.init and while loop)
        code1 = """import pygame
pygame.init()
while running:
    pass
"""
        assert executor._is_main_script(code1) is True

        # Should be main script (has __name__ check)
        code2 = """def main():
    pass

if __name__ == "__main__":
    main()
"""
        assert executor._is_main_script(code2) is True

        # Should NOT be main script (only one indicator)
        code3 = """import pygame
print("hello")
"""
        assert executor._is_main_script(code3) is False
