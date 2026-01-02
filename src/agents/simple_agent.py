from langchain.agents import create_agent
from src.tools import create_doc_getter,  validate_ISA_schema
from src.model import get_chat_model
from src.prompts.registry import PromptRegistry
from src.state.v2 import ISAExtractionStateV1
from src.document_processing import DocumentManager
from src.prompts.renderer import render_prompt
from src.utils import log
from langchain.messages import SystemMessage

def get_simple_agent_v1(doc_manager: DocumentManager):
    model = get_chat_model()

    prompt_registry = PromptRegistry()
    prompt : SystemMessage = render_prompt(prompt_registry.get_prompt("simple_agent", "v1"), {})

    doc_chunk_tool = create_doc_getter(doc_manager)

    agent_name = "simple_agent_v1"

    agent = create_agent(
        model = model, 
        tools = [doc_chunk_tool],
        name = agent_name,
        system_prompt = prompt,
        state_schema = ISAExtractionStateV1
    )

    log(f"Finished creating agent {agent_name}")

    return agent


