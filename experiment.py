import os
from enum import Enum
from dotenv import load_dotenv
from litellm import completion 

class Provider(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

def get_api_key(provider: Provider) -> str:
    """Retrieve API key from environment variables."""

    load_dotenv()

    if provider == Provider.OPENAI:
        return os.getenv("OPENAI_API_KEY")
    elif provider == Provider.HUGGINGFACE:
        return os.getenv("HUGGINGFACE_API_KEY")
    else:
        raise ValueError("Unsupported provider")
    

def query_model(provider: Provider, model: str, prompt: str) -> str:
    """Basic template to query a language model from a specified provider.

    provider: Provider enum indicating which LLM provider to use.
    model: The model name to query.
    prompt: The prompt to send to the model.
    """

    api_key = get_api_key(provider)

    if not api_key:
        raise ValueError(f"API key for {provider.value} not found in environment variables.")
    
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    print(response)

if __name__ == "__main__":
    query_model(Provider.HUGGINGFACE, "huggingface/WizardLM/WizardCoder-Python-34B-V1.0", "What are you?")
    