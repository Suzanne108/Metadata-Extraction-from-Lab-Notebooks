

from src.state.v2 import ISAExtractionStateV2
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.utils import DraftUpdate

def create_ISA_updater(model: ChatOpenAI, system_prompt: str, inference_prompt_template: str):

    def ISA_updater(state: ISAExtractionStateV2) -> dict:

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

        response : DraftUpdate = model.invoke(messages)

        isa_updater_message = f"Updated draft ISA metadata. New draft: {response.updated_draft}, message for planner: {response.message}"

        return {
            "draft_isa_metadata": response.updated_draft,
            "messages": state.get("messages", []) +  [AIMessage(content=isa_updater_message)]
        }


    
    return ISA_updater


