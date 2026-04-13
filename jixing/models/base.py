from abc import ABC, abstractmethod
from typing import Any


class ModelAdapterBase(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> tuple[str, dict[str, Any]]:
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> tuple[str, dict[str, Any]]:
        pass

    def get_config(self) -> dict:
        return self.config.copy()
