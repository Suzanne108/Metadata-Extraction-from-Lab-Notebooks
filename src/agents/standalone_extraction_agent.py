from langchain.agents import create_agent
from src.tools import create_doc_getter, create_schema_getter, validate_ISA_schema
from src.model import get_chat_model
from src.prompts.registry import PromptRegistry
from src.state import ISAExtractionState
from src.document_processing import DocumentManager
from src.prompts.renderer import render_prompt
from src.utils import log
from langchain.messages import SystemMessage

def get_standalone_extraction_agent(doc_manager: DocumentManager):
    model = get_chat_model()

    prompt_registry = PromptRegistry()
    prompt : SystemMessage = render_prompt(prompt_registry.get_prompt("extract_ISA", "v1"), {})

    doc_chunk_tool = create_doc_getter(doc_manager)
    schema_getter_tools = create_schema_getter()

    agent_name = "standalone_extraction_agent"

    agent = create_agent(
        model = model, 
        tools = [doc_chunk_tool, validate_ISA_schema] + schema_getter_tools,
        name = agent_name,
        system_prompt = prompt,
        state_schema = ISAExtractionState
    )

    log(f"Finished creating agent {agent_name}")

    return agent


