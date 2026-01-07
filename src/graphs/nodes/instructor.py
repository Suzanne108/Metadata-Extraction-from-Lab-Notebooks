from src.state.v2 import ISAExtractionStateV2
from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt

def instructor(state: ISAExtractionStateV2):

    prompt_registry = PromptRegistry()
    prompt : SystemMessage = SystemMessage(render_prompt(prompt_registry.get_prompt("instructor", "v1"), {}))

    return {
            "messages": state.get("messages", []) + [prompt]
    }
    