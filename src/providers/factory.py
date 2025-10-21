from .base_provider import Provider, LLMProvider
from .openai_provider import OpenAIProvider
from dotenv import load_dotenv
import os

class LLMProviderFactory():

    def __init__(self):
        self.provider = None
        self.model = None

    def set_provider(self, provider: Provider):
        self.provider = provider

    def set_model(self, model: str):
        self.model = model  

    def get_provider(self) -> LLMProvider:
        """Factory function to get the appropriate LLM provider instance."""

        if self.model is None:
            raise ValueError("Model must be set before getting provider.")

        if self.provider == Provider.OPENAI:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            return OpenAIProvider(api_key, self.model)
        else:
            raise ValueError("Unsupported provider or provider not set.")