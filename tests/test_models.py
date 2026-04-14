from unittest.mock import MagicMock, patch

import pytest

from jixing.models.base import ModelAdapterBase
from jixing.models.moxing import MoxingModel
from jixing.models.ollama import OllamaModel


class TestModelAdapterBase:
    def test_init(self):
        class DummyAdapter(ModelAdapterBase):
            def chat(self, messages, **kwargs):
                return "", {}

            def generate(self, prompt, **kwargs):
                return "", {}

        adapter = DummyAdapter("test-model", temperature=0.7)
        assert adapter.model_name == "test-model"
        assert adapter.config == {"temperature": 0.7}

    def test_get_config(self):
        class DummyAdapter(ModelAdapterBase):
            def chat(self, messages, **kwargs):
                return "", {}

            def generate(self, prompt, **kwargs):
                return "", {}

        adapter = DummyAdapter("test-model", key1="value1", key2=42)
        config = adapter.get_config()
        assert config == {"key1": "value1", "key2": 42}

    def test_get_config_returns_copy(self):
        class DummyAdapter(ModelAdapterBase):
            def chat(self, messages, **kwargs):
                return "", {}

            def generate(self, prompt, **kwargs):
                return "", {}

        adapter = DummyAdapter("test-model", key="value")
        config = adapter.get_config()
        config["key"] = "modified"
        assert adapter.get_config()["key"] == "value"

    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            ModelAdapterBase("test")


class TestOllamaModel:
    def test_init(self):
        model = OllamaModel("gemma3:1b")
        assert model.model_name == "gemma3:1b"
        assert "http://localhost:11434" in model.base_url

    def test_init_custom_url(self):
        model = OllamaModel("gemma3:1b", base_url="http://custom:11434")
        assert model.base_url == "http://custom:11434"

    def test_init_strips_trailing_slash(self):
        model = OllamaModel("gemma3:1b", base_url="http://localhost:11434/")
        assert model.base_url == "http://localhost:11434"

    def test_init_with_kwargs(self):
        model = OllamaModel("gemma3:1b", temperature=0.5, num_ctx=4096)
        assert model.config["temperature"] == 0.5
        assert model.config["num_ctx"] == 4096

    @patch("requests.post")
    def test_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Hello!"},
            "eval_count": 50,
            "eval_duration": 4000000000,
            "prompt_eval_count": 20,
            "total_duration": 5000000000,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = OllamaModel("gemma3:1b")
        content, metrics = model.chat([{"role": "user", "content": "Hello"}])

        assert content == "Hello!"
        assert metrics["provider"] == "ollama"
        assert metrics["model"] == "gemma3:1b"
        assert metrics["eval_count"] == 50

    @patch("requests.post")
    def test_generate(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "Generated text",
            "eval_count": 30,
            "eval_duration": 3000000000,
            "load_duration": 500000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100000000,
            "total_duration": 4000000000,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = OllamaModel("gemma3:1b")
        content, metrics = model.generate("Generate something")

        assert content == "Generated text"
        assert metrics["eval_count"] == 30
        assert metrics["load_duration"] == 500000000

    @patch("requests.post")
    def test_chat_with_kwargs(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Response"},
            "eval_count": 10,
            "eval_duration": 1000000000,
            "prompt_eval_count": 5,
            "total_duration": 2000000000,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = OllamaModel("gemma3:1b")
        content, metrics = model.chat(
            [{"role": "user", "content": "Hi"}],
            temperature=0.5,
            num_predict=100,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["temperature"] == 0.5
        assert payload["num_predict"] == 100

    @patch("requests.post")
    def test_chat_error_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Connection error")
        mock_post.return_value = mock_resp

        model = OllamaModel("gemma3:1b")
        with pytest.raises(Exception):
            model.chat([{"role": "user", "content": "Hello"}])


class TestMoxingModel:
    def test_init(self):
        model = MoxingModel("test-model")
        assert model.model_name == "test-model"
        assert "http://localhost:8080" in model.base_url

    def test_init_custom_url(self):
        model = MoxingModel("test-model", base_url="http://custom:8080")
        assert model.base_url == "http://custom:8080"

    def test_init_strips_trailing_slash(self):
        model = MoxingModel("test-model", base_url="http://localhost:8080/")
        assert model.base_url == "http://localhost:8080"

    def test_init_with_kwargs(self):
        model = MoxingModel("test-model", temperature=0.8)
        assert model.config["temperature"] == 0.8

    @patch("requests.post")
    def test_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Moxing response!"}}],
            "usage": {"completion_tokens": 30},
            "latency_ms": 250,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = MoxingModel("test-model")
        content, metrics = model.chat([{"role": "user", "content": "Hello"}])

        assert content == "Moxing response!"
        assert metrics["provider"] == "moxing"
        assert metrics["tokens"] == 30
        assert metrics["latency_ms"] == 250

    @patch("requests.post")
    def test_generate(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Generated!"}}],
            "usage": {"completion_tokens": 20},
            "latency_ms": 150,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = MoxingModel("test-model")
        content, metrics = model.generate("Generate text")

        assert content == "Generated!"
        assert metrics["tokens"] == 20

    @patch("requests.post")
    def test_chat_with_kwargs(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"completion_tokens": 10},
            "latency_ms": 100,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = MoxingModel("test-model")
        model.chat(
            [{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=200,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 200

    @patch("requests.post")
    def test_chat_error_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Server error")
        mock_post.return_value = mock_resp

        model = MoxingModel("test-model")
        with pytest.raises(Exception):
            model.chat([{"role": "user", "content": "Hello"}])

    @patch("requests.post")
    def test_chat_headers(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"completion_tokens": 5},
            "latency_ms": 50,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = MoxingModel("test-model")
        model.chat([{"role": "user", "content": "Hi"}])

        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Content-Type"] == "application/json"
