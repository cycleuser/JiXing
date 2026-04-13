from typing import Any

import requests

from .base import ModelAdapterBase


class MoxingModel(ModelAdapterBase):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8080", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict], **kwargs) -> tuple[str, dict[str, Any]]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": messages,
        }
        payload.update(kwargs)

        response = requests.post(
            url, json=payload, headers=headers, timeout=kwargs.get("timeout", 300)
        )
        response.raise_for_status()

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        metrics = {
            "provider": "moxing",
            "model": self.model_name,
            "tokens": data.get("usage", {}).get("completion_tokens", 0),
            "latency_ms": data.get("latency_ms", 0),
        }

        return content, metrics

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict[str, Any]]:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
