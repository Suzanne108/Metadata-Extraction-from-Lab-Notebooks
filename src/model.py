"module providing llm model configuration"

from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os

# TODO: this should be changed so that you can centrally for the workflow determine which components should be using which model. For example, the planner should use a different model than the specialist agents. This should be configurable in a central place, rather than hardcoded in this file.
def get_chat_model():
    """"Returns basic LLM model configuration."""

    #TODO: change this to not be dependent on the file structure
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    open_ai_key = os.getenv("OPENAI_API_KEY")

    return ChatOpenAI(model="gpt-5-nano", temperature=0, api_key=open_ai_key)

def get_embedding_model():
    """Returns basic embedding model configuration."""

    #TODO: change this to not be dependent on the file structure
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    open_ai_key = os.getenv("OPENAI_API_KEY")

    return OpenAIEmbeddings(model="text-embedding-3-large", api_key=open_ai_key)
    