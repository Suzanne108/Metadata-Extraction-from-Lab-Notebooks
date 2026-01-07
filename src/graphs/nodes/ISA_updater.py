

from src.state.v2 import ISAExtractionStateV2
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.utils import DraftUpdate

def create_ISA_updater(model: ChatOpenAI, system_prompt: str):

    #TODO: refactor to accept prompt as argument

    def ISA_updater(state: ISAExtractionStateV2, prompt_template: str ) -> dict:

        draft_isa_metadata = state.get("draft_isa_metadata", {})
        instruction = state['orchestrator_decision'].instruction
        latest_evidence = state.get("latest_evidence", [])
        
        registry = PromptRegistry()
        prompt_template = registry.get_prompt("ISA_updater", "v1")  
        prompt = render_prompt(
            prompt_template,
            {
                "draft_isa_metadata": draft_isa_metadata,
                "instruction": instruction,
                "latest_evidence": latest_evidence
            }
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        response : DraftUpdate = model.invoke(prompt)

        return {
            "draft_isa_metadata": response.updated_draft,
            "messages": state.get("messages", []) +  [AIMessage(content=response.message)]
        }


    
    return ISA_updater


