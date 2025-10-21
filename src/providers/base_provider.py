from enum import Enum
from abc import ABC, abstractmethod

class Provider(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def set_model(self, model: str):
        """Set the model to be used by the provider."""
        self.model = model

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Query the language model with the given prompt."""
        pass