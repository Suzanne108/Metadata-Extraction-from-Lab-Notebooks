

import json
from src.state.v3 import ISAExtractionStateV3
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.utils import invoke_model_with_tools
from typing import List

def create_ISA_updater(model: ChatOpenAI, schema_getter_tools: List, system_prompt: str, inference_prompt_template: str):

    def ISA_updater(state: ISAExtractionStateV3) -> dict:

        draft_isa_metadata = state.get("draft_isa_metadata", {})
        instruction = state['orchestrator_decision'].instruction
        latest_evidence = state.get("latest_evidence", [])
        
        prompt = render_prompt(
            inference_prompt_template,
            {
                "draft_isa_metadata": draft_isa_metadata,
                "instruction": instruction,
                "latest_evidence": latest_evidence
            }
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        # Invoke model with tool-calling support
        response = invoke_model_with_tools(model, messages, schema_getter_tools, max_iterations=10)
        
        print("ISA updater response:", response.content)

        # Parse the response and update the state
        try:
            isa_updated_draft = json.loads(response.content)
            isa_updater_message = f"NODE: UPDATE_DRAFT; status: success; instruction: {instruction}"
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response content: {response.content}")
            # Keep the old draft if parsing fails
            isa_updated_draft = draft_isa_metadata
            isa_updater_message = f"NODE: UPDATE_DRAFT; status: error; error: failed to parse updated draft: {str(e)}; instruction: {instruction}"

        return {
            "draft_isa_metadata": isa_updated_draft,
            "messages": state.get("messages", []) +  [AIMessage(content=isa_updater_message)]
        }


    
    return ISA_updater


