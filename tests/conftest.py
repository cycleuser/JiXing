import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def tmp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def mock_ollama_response():
    return {
        "model": "gemma3:1b",
        "created_at": "2024-01-01T00:00:00.000000Z",
        "response": "Hello! I'm doing well, thank you for asking.",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 5000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 100000000,
        "eval_count": 50,
        "eval_duration": 4000000000,
    }


@pytest.fixture
def mock_moxing_response():
    return {
        "result": {"text": "Hello from Moxing!"},
        "usage": {"completion_tokens": 30, "prompt_tokens": 15},
        "latency_ms": 250,
    }


@pytest.fixture
def mock_requests_post(mock_ollama_response, mock_moxing_response):
    def _mock_post(url, *args, **kwargs):
        resp = MagicMock()
        if "11434" in url:
            resp.json.return_value = mock_ollama_response
        elif "8080" in url:
            resp.json.return_value = mock_moxing_response
        else:
            resp.json.return_value = {}
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        return resp

    return _mock_post


@pytest.fixture
def sample_session_data():
    return {
        "id": "test-session-001",
        "model_provider": "ollama",
        "model_name": "gemma3:1b",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:01:00+00:00",
        "system_info": {
            "hostname": "test-host",
            "os_system": "Linux",
            "python_version": "3.10.0",
        },
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": "2024-01-01T00:00:30+00:00",
                "metrics": {"eval_count": 20},
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you!",
                "timestamp": "2024-01-01T00:01:00+00:00",
                "metrics": {"eval_count": 50},
            },
        ],
        "total_tokens": 70,
        "total_duration_ms": 4500,
    }


@pytest.fixture
def sample_memory_data():
    return {
        "memory_id": "mem-001",
        "session_id": "sess-001",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "content": "This is a test memory content with some important information about Python programming and software architecture.",
        "content_hash": "abc123def456",
        "semantic_vector": np.random.rand(128).astype(np.float32),
        "spatial_context": {
            "device_id": "dev-001",
            "hostname": "test-host",
            "os_info": "Linux 5.15",
        },
        "temporal_links": [],
        "spatial_links": [],
        "semantic_similarity": [],
        "compression_level": 1,
        "importance_score": 0.85,
        "access_count": 3,
        "last_accessed": "2024-01-01T00:05:00+00:00",
        "metadata": {"source": "test", "has_code": True},
    }


@pytest.fixture
def multiple_memories():
    memories = []
    for i in range(10):
        memories.append(
            {
                "memory_id": f"mem-{i:03d}",
                "session_id": f"sess-{i % 3:03d}",
                "timestamp": f"2024-01-01T{0:02d}:{i * 5:02d}:00+00:00",
                "content": f"Memory content number {i} about topic {'Python' if i % 2 == 0 else 'AI'}",
                "content_hash": f"hash{i:04d}",
                "semantic_vector": None,
                "spatial_context": {"device_id": f"dev-{i % 2:03d}"},
                "temporal_links": [],
                "spatial_links": [],
                "semantic_similarity": [],
                "compression_level": 1,
                "importance_score": 0.5 + (i * 0.05),
                "access_count": i + 1,
                "last_accessed": f"2024-01-01T00:{i * 5:02d}:00+00:00",
                "metadata": {},
            }
        )
    return memories


@pytest.fixture
def sample_tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input string"},
                    },
                    "required": ["input"],
                },
            },
        }
    ]


@pytest.fixture
def cli_args_ollama():
    args = MagicMock()
    args.command = "ollama"
    args.subcommand = "run"
    args.model = "gemma3:1b"
    args.prompt = ["Hello", "world"]
    args.session = None
    args.interactive = False
    args.json_output = False
    args.verbose = False
    args.quiet = False
    args.output = None
    return args


@pytest.fixture
def cli_args_sessions_list():
    args = MagicMock()
    args.command = "sessions"
    args.subcommand = "list"
    args.provider = None
    args.model = None
    args.limit = 100
    args.json_output = False
    return args


@pytest.fixture
def cli_args_stats():
    args = MagicMock()
    args.command = "stats"
    args.json_output = False
    return args


@pytest.fixture
def cli_args_info():
    args = MagicMock()
    args.command = "info"
    args.json_output = False
    return args


@pytest.fixture
def cli_args_search():
    args = MagicMock()
    args.command = "search"
    args.query = "Python"
    args.session = None
    args.provider = None
    args.limit = 100
    args.json_output = False
    return args
