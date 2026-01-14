from src.state.v3 import ISAExtractionStateV3
from src.graphs.utils import Action, Node


def routing_function(state: ISAExtractionStateV3) -> str:
    """Map the planner's enum action to a concrete graph node label.

    Using explicit enum mapping avoids brittle string matching that can
    silently fail to route when labels change or differ in whitespace/casing.
    """
    decision = state.get("orchestrator_decision")
    if decision is None or getattr(decision, "action", None) is None:
        # Fallback to planner loop if we somehow have no decision
        return Node.PLANNER_AGENT.value

    action: Action = decision.action

    if action == Action.UPDATE_DRAFT:
        return Node.UPDATE_DRAFT_AGENT.value
    if action == Action.SEARCH_DOCUMENT:
        return Node.SEARCH_DOCUMENT.value
    if action == Action.VALIDATE:
        return Node.VALIDATE.value
    if action == Action.FINISH_TASK:
        return Node.TASK_PROGRESS.value
    if action == Action.STOP:
        return Node.STOP.value

    # Default: keep planner in control
    return Node.PLANNER_AGENT.value