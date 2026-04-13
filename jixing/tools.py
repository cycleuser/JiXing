import json
from typing import Any

from .api import (
    delete_session,
    get_session,
    get_stats,
    query_sessions,
    run_moxing,
    run_ollama,
    search_messages,
)

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "jixing_run_ollama",
            "description": "Run an Ollama model with a prompt. Records interaction with metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g., gemma3:1b, llama2, mistral)",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the model",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID to continue a previous conversation",
                    },
                },
                "required": ["model", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jixing_run_moxing",
            "description": "Run a Moxing model with a prompt. Moxing is a GGUF model runner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name or GGUF file path",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the model",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID to continue a previous conversation",
                    },
                },
                "required": ["model", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jixing_query_sessions",
            "description": "List all recorded sessions with optional filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_provider": {
                        "type": "string",
                        "description": "Filter by provider (ollama or moxing)",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Filter by model name (partial match)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 100,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jixing_get_session",
            "description": "Get detailed information about a specific session",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session ID to retrieve",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jixing_delete_session",
            "description": "Delete a session and all its messages",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session ID to delete",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jixing_search_messages",
            "description": "Search through all messages in the long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID to search within",
                    },
                    "model_provider": {
                        "type": "string",
                        "description": "Filter by provider (ollama or moxing)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 100,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jixing_get_stats",
            "description": "Get usage statistics for all sessions",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any] | str) -> dict:
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "jixing_run_ollama":
        result = run_ollama(**arguments)
        return result.to_dict()

    if name == "jixing_run_moxing":
        result = run_moxing(**arguments)
        return result.to_dict()

    if name == "jixing_query_sessions":
        result = query_sessions(**arguments)
        return result.to_dict()

    if name == "jixing_get_session":
        result = get_session(**arguments)
        return result.to_dict()

    if name == "jixing_delete_session":
        result = delete_session(**arguments)
        return result.to_dict()

    if name == "jixing_search_messages":
        result = search_messages(**arguments)
        return result.to_dict()

    if name == "jixing_get_stats":
        result = get_stats()
        return result.to_dict()

    raise ValueError(f"Unknown tool: {name}")
