from src.graphs.utils import NextStep
from src.state.v2 import ISAExtractionStateV2
from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import json

def create_planner(model: ChatOpenAI):

    def create_planner_message(response: NextStep) -> AIMessage:
        "Creates an AIMessage from the NextStep response"
        
        content = f"Next Action: {response.action.value}\n"
        if response.focus:
            content += f"Focus: {response.focus}\n"
        if response.query:
            content += f"Query: {response.query}\n"
        if response.k:
            content += f"K: {response.k}\n"
        if response.schema:
            content += f"Schema: {response.schema}\n"
        content += f"Rationale: {response.rationale}\n"

        return AIMessage(content=content)


    def planner(state: ISAExtractionStateV2) -> dict:
        "Decides next action based on the current state"

        vr = state.get("validation_result") or {}
        draft = state.get("draft_isa_metadata") or {}

        # Call LLM to analyze state and decide next step

        #TODO: update the prompt so that the arguments that should be given are also explained
        prompt = f"""
        Current ISA Draft: {json.dumps(draft, indent=2)}
        
        Latest Validation Result: {json.dumps(vr, indent=2)}
        
        Schema Cache: {list(state.get("schema_cache", {}).keys())}
        
        Evidence found so far: {json.dumps(
            {k: len(v) for k, v in state.get("evidence", {}).items()},
            indent=2
        )}
        
        Decide the next action:
        - "READ_SCHEMA" if you need to understand schema requirements for a field
        - "SEARCH_DOCUMENT" if you need to find evidence from the document
        - "UPDATE_DRAFT" if you can fix the draft deterministically
        - "VALIDATE" if the draft is complete and ready for validation
        - "end" if the extraction is done

        """
        # Is of type NextStep
        response = model.invoke(prompt)

        return {
            "messages": state.get("messages", []) + [create_planner_message(response)],
            "orchestrator_decision": response
        }

    return planner