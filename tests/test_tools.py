import json
from unittest.mock import MagicMock, patch

import pytest

from jixing.tools import TOOLS, dispatch


class TestToolsSchema:
    def test_tools_is_list(self):
        assert isinstance(TOOLS, list)
        assert len(TOOLS) >= 1

    def test_tool_names(self):
        for tool in TOOLS:
            name = tool["function"]["name"]
            assert name.startswith("jixing_")

    def test_tool_structure(self):
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
        for tool in TOOLS:
            func = tool["function"]
            props = func["parameters"]["properties"]
            for req in func["parameters"]["required"]:
                assert req in props, f"Required '{req}' not in properties"

    def test_specific_tool_names_exist(self):
        tool_names = {tool["function"]["name"] for tool in TOOLS}
        expected_names = {
            "jixing_run_ollama",
            "jixing_run_moxing",
            "jixing_query_sessions",
            "jixing_get_session",
            "jixing_delete_session",
            "jixing_search_messages",
            "jixing_get_stats",
        }
        assert expected_names.issubset(tool_names)

    def test_ollama_tool_has_required_fields(self):
        ollama_tool = next(t for t in TOOLS if t["function"]["name"] == "jixing_run_ollama")
        props = ollama_tool["function"]["parameters"]["properties"]
        assert "model" in props
        assert "prompt" in props
        assert ollama_tool["function"]["parameters"]["required"] == ["model", "prompt"]

    def test_moxing_tool_has_required_fields(self):
        moxing_tool = next(t for t in TOOLS if t["function"]["name"] == "jixing_run_moxing")
        props = moxing_tool["function"]["parameters"]["properties"]
        assert "model" in props
        assert "prompt" in props

    def test_search_tool_has_required_fields(self):
        search_tool = next(t for t in TOOLS if t["function"]["name"] == "jixing_search_messages")
        assert search_tool["function"]["parameters"]["required"] == ["query"]

    def test_stats_tool_has_no_required_fields(self):
        stats_tool = next(t for t in TOOLS if t["function"]["name"] == "jixing_get_stats")
        assert stats_tool["function"]["parameters"]["required"] == []

    def test_tool_descriptions_are_meaningful(self):
        for tool in TOOLS:
            desc = tool["function"]["description"]
            assert len(desc) > 10


class TestToolsDispatch:
    def test_dispatch_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("nonexistent_tool", {})

    def test_dispatch_json_string_args(self):
        args = json.dumps({"query": "test"})
        result = dispatch("jixing_search_messages", args)
        assert isinstance(result, dict)
        assert "success" in result

    def test_dispatch_dict_args(self):
        result = dispatch("jixing_get_stats", {})
        assert "success" in result

    @patch("jixing.tools.run_ollama")
    def test_dispatch_run_ollama(self, mock_run):
        mock_run.return_value = MagicMock(success=True, to_dict=lambda: {"success": True})
        args = json.dumps({"model": "gemma3:1b", "prompt": "Hello"})
        result = dispatch("jixing_run_ollama", args)
        assert "success" in result
        mock_run.assert_called_once_with(model="gemma3:1b", prompt="Hello")

    @patch("jixing.tools.run_moxing")
    def test_dispatch_run_moxing(self, mock_run):
        mock_run.return_value = MagicMock(success=True, to_dict=lambda: {"success": True})
        args = json.dumps({"model": "test-model", "prompt": "Hello"})
        result = dispatch("jixing_run_moxing", args)
        assert "success" in result
        mock_run.assert_called_once_with(model="test-model", prompt="Hello")

    def test_dispatch_query_sessions(self):
        args = json.dumps({"limit": 5})
        result = dispatch("jixing_query_sessions", args)
        assert "success" in result

    def test_dispatch_get_session(self):
        result = dispatch("jixing_get_session", {"session_id": "nonexistent"})
        assert "success" in result

    def test_dispatch_delete_session(self):
        result = dispatch("jixing_delete_session", {"session_id": "nonexistent"})
        assert "success" in result

    def test_dispatch_search_messages(self):
        result = dispatch("jixing_search_messages", {"query": "test"})
        assert "success" in result

    def test_dispatch_get_stats(self):
        result = dispatch("jixing_get_stats", {})
        assert "success" in result
        assert result["success"] is True

    def test_dispatch_with_empty_string_args(self):
        with pytest.raises(json.JSONDecodeError):
            dispatch("jixing_get_stats", "")

    def test_dispatch_with_invalid_json_args(self):
        with pytest.raises(json.JSONDecodeError):
            dispatch("jixing_get_stats", "{invalid json}")
