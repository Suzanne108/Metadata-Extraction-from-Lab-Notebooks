"module providing llm model configuration"

from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os

def get_chat_model(model_name: str = "gpt-5-nano", reasoning_effort: str = "medium", max_retries: int = 3) -> ChatOpenAI:
    """"Returns basic LLM model configuration."""

    #TODO: change this to not be dependent on the file structure
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    open_ai_key = os.getenv("OPENAI_API_KEY")

    return ChatOpenAI(model=model_name, temperature=0, api_key=open_ai_key, reasoning_effort=reasoning_effort, max_retries=max_retries)

def get_embedding_model():
    """Returns basic embedding model configuration."""

    #TODO: change this to not be dependent on the file structure
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    open_ai_key = os.getenv("OPENAI_API_KEY")

    return OpenAIEmbeddings(model="text-embedding-3-large", api_key=open_ai_key)
    