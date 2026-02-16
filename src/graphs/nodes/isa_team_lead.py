import json
from typing import List

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt
from src.graphs.types.TeamLeadDecision import TeamLeadDecision


def create_isa_team_lead(model: ChatOpenAI, system_prompt: str, inference_prompt_template) -> callable:
    """Creates a team lead node that routes planner tasks to specialized creators.

    The model should return a strict JSON object like:
    {"target": "investigation" | "publication" | "person", "subtask": "...", "reason": "..."}
    """

    def isa_team_lead(state: ISAExtractionState) -> dict:
        instruction = state.get("orchestrator_decision").instruction 
        latest_evidence = state.get("latest_evidence") or []

        prompt = render_prompt(
            inference_prompt_template,
            {
                "instruction": instruction,
                "latest_evidence": latest_evidence
            },
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response = model.invoke(messages)

        print(response)

        return {
            "team_lead_decision": response,
            "messages": [AIMessage(content=create_message(response))],
            "last_action": "TEAM_LEAD_ROUTE",
        }
    
    def create_message(decision: TeamLeadDecision) -> str:
        return f"Routed to {decision.target} specialist \n for subtask: {decision.instruction}. \n Reason: {decision.rationale}"

    return isa_team_lead
