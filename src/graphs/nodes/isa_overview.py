from src.state.v3 import ISAExtractionStateV3
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.utils import invoke_model_with_tools


def create_isa_overview(model: ChatOpenAI, schema_getter_tools: list, system_prompt: str, inference_prompt_template: str):

    def isa_overview(state: ISAExtractionStateV3) -> dict:
        prompt = render_prompt(inference_prompt_template, {})
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        response = invoke_model_with_tools(model, messages, schema_getter_tools, max_iterations=20)

        mental_model = response.content if hasattr(response, "content") else str(response)
        return {
            "isa_mental_model": mental_model,
            "messages": state.get("messages", []) + [AIMessage(content=mental_model)]
        }

    return isa_overview
