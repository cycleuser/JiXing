from typing import Any

import requests

from .base import ModelAdapterBase


class OllamaModel(ModelAdapterBase):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict], **kwargs) -> tuple[str, dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        payload.update(kwargs)

        response = requests.post(url, json=payload, timeout=kwargs.get("timeout", 300))
        response.raise_for_status()

        data = response.json()
        content = data.get("message", {}).get("content", "")

        metrics = {
            "provider": "ollama",
            "model": self.model_name,
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "total_duration": data.get("total_duration", 0),
        }

        return content, metrics

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict[str, Any]]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        payload.update(kwargs)

        response = requests.post(url, json=payload, timeout=kwargs.get("timeout", 300))
        response.raise_for_status()

        data = response.json()
        content = data.get("response", "")

        metrics = {
            "provider": "ollama",
            "model": self.model_name,
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
            "load_duration": data.get("load_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
            "total_duration": data.get("total_duration", 0),
        }

        return content, metrics
