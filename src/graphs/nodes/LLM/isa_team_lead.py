from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from src.graphs.nodes.base_node import LLMGraphNode
from src.graphs.types.Node import GraphComponentId
from src.graphs.types.Action import WorkflowAction
from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt
from src.graphs.types.TeamLeadDecision import TeamLeadDecision


class ISATeamLeadNode(LLMGraphNode):
    """LLM node that routes planner tasks to specialized creators.

    The model should return a strict JSON object like:
    {"target": "investigation" | "publication" | "person", "subtask": "...", "reason": "..."}
    """

    def __init__(self, model: ChatOpenAI, system_prompt: str, inference_prompt_template) -> None:
        super().__init__(
            node_id=GraphComponentId.TEAM_LEAD.value,
            model=model,
            system_prompt=system_prompt,
            inference_prompt_template=inference_prompt_template,
        )

    def _create_message(self, decision: TeamLeadDecision) -> str:
        return f"Routed to {decision.target} specialist \n for subtask: {decision.instruction}. \n Reason: {decision.rationale}"

    def run(self, state: ISAExtractionState) -> dict:
        instruction = state.get("planner_decision").instruction 
        latest_evidence = state.get("latest_evidence") or []

        prompt = render_prompt(
            self.inference_prompt_template,
            {
                "instruction": instruction,
                "latest_evidence": latest_evidence
            },
        )

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        response = self.model.invoke(messages)

        print(response)

        return {
            "team_lead_decision": response,
            "messages": [AIMessage(content=self._create_message(response))],
            "last_action": WorkflowAction.TEAM_LEAD_ROUTE,
        }

