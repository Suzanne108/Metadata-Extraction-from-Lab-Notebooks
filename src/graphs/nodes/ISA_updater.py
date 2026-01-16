

import json
from src.state import ISAExtractionState
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from src.prompts.renderer import render_prompt
from src.graphs.utils import invoke_model_with_tools
from typing import List

def _enforce_investigation_skeleton(draft: dict) -> dict:
    """Ensure the draft is a non-empty Investigation object with allowed properties only.

    - Adds '@type': 'Investigation' if missing
    - Ensures array fields exist (empty arrays) to keep object truthy
    - Leaves existing allowed fields untouched
    """
    if not isinstance(draft, dict):
        draft = {}

    # Minimal truthy structure compliant with investigation schema properties
    draft.setdefault("@type", "Investigation")
    for key in [
        "ontologySourceReferences",
        "publications",
        "people",
        "studies",
        "comments",
    ]:
        if key not in draft:
            draft[key] = []

    return draft

def create_ISA_updater(model: ChatOpenAI, schema_getter_tools: List, system_prompt: str, inference_prompt_template: str):

    def ISA_updater(state: ISAExtractionState) -> dict:

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
        response = invoke_model_with_tools(model, messages, schema_getter_tools, max_iterations=20)

        # Parse the response and update the state
        try:
            isa_updated_draft = json.loads(response.content)
            # Enforce a non-empty, schema-shaped investigation skeleton
            isa_updated_draft = _enforce_investigation_skeleton(isa_updated_draft)
            isa_updater_message = "NODE: UPDATE_DRAFT; status: success; normalization: skeleton_enforced"
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response content: {response.content}")
            # Keep the old draft if parsing fails
            isa_updated_draft = _enforce_investigation_skeleton(draft_isa_metadata)
            isa_updater_message = "NODE: UPDATE_DRAFT; status: error; normalization: skeleton_enforced; error: failed to parse updated draft: " + str(e)

        return {
            "draft_isa_metadata": isa_updated_draft,
            "messages": [AIMessage(content=isa_updater_message), AIMessage(content=response.content)],
            "last_action": "UPDATE_DRAFT"
        }


    
    return ISA_updater


