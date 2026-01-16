from typing import List
from xml.dom.minidom import Document
from src.state.v3 import ISAExtractionStateV3
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.utils import invoke_model_with_tools


def create_isa_overview(model: ChatOpenAI, schema_getter_tools: list, system_prompt: str, inference_prompt_template: str, isa_overview_doc: List[Document]):

    def isa_overview(state: ISAExtractionStateV3) -> dict:
        full_text = ""
        try:
            if isinstance(isa_overview_doc, list):
                full_text = "\n\n".join([getattr(p, "page_content", str(p)) for p in isa_overview_doc])
            else:
                full_text = getattr(isa_overview_doc, "page_content", str(isa_overview_doc))
        except Exception:
            full_text = str(isa_overview_doc)

        prompt = render_prompt(inference_prompt_template, {"ISA_abstract_model": full_text})
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        response = invoke_model_with_tools(model, messages, schema_getter_tools, max_iterations=20)

        mental_model = response.content if hasattr(response, "content") else str(response)
        return {
            "isa_mental_model": mental_model,
            "messages": [AIMessage(content=mental_model)]
        }

    return isa_overview
