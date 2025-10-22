from .base_provider import LLMProvider
from litellm import completion
import os

class HuggingFaceAPIProvider(LLMProvider):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)

        os.environ["HUGGINGFACE_API_KEY"] = api_key

    def set_huggingface_provider(self, provider: str):
        """Set the provider to HuggingFace API."""
        self.provider = provider

    def query(self, prompt: str) -> str:
        """Query the HuggingFace model with the given prompt."""

        if not self.provider:
            raise ValueError("Provider not set. Please set the provider before querying.")
        if not self.model:
            raise ValueError("Model not set. Please set the model before querying.")

        response = completion(
            model="huggingface/" + self.provider + "/" + self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response