from .base_provider import LLMProvider
from litellm import completion
import os

class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)

        os.environ["OPENAI_API_KEY"] = api_key

    def query(self, prompt: str) -> str:
        """Query the OpenAI model with the given prompt."""

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response