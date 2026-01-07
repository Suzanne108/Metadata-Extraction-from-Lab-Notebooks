

from src.state.v2 import ISAExtractionStateV2
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt

def create_ISA_updater(model: ChatOpenAI):

    def ISA_updater(state: ISAExtractionStateV2):

        draft_isa_metadata = state.get("draft_isa_metadata", {})
        focus = state['orchestrator_decision'].focus
        latest_evidence = state.get("latest_evidence", [])
        
        registry = PromptRegistry()
        prompt_template = registry.get_prompt("ISA_updater", "v1")  
        prompt = render_prompt(
            prompt_template,
            {
                "draft_isa_metadata": draft_isa_metadata,
                "focus": focus,
                "latest_evidence": latest_evidence
            }
        )
        
        response = model.invoke(prompt)

        return {
            "draft_isa_metadata": response.content,
            "messages": state.get("messages", []) +  [AIMessage(content="Updated draft ISA metadata.")]
        }


    
    return ISA_updater


