from src.state import ISAExtractionState
from src.graphs.types.Action import Action
from src.graphs.types.Node import Node


def routing_function(state: ISAExtractionState) -> str:
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
        # Route to team lead for specialized creation routing
        return Node.TEAM_LEAD.value
    if action == Action.SEARCH_DOCUMENT:
        return Node.SEARCH_DOCUMENT.value
    if action == Action.FINISH_TASK:
        return Node.TASK_PROGRESS.value
    if action == Action.STOP:
        # When planner signals completion, first run assembly advice before assembly/export
        return Node.ASSEMBLY_ADVISOR.value

    # Default: keep planner in control
    return Node.PLANNER_AGENT.value


def team_lead_routing(state: ISAExtractionState) -> str:
    """Route based on team lead decision stored in state."""
    decision = state.get("team_lead_decision")
    target = getattr(decision, "target", None)
    if target == "investigation":
        return Node.INVESTIGATION_CREATOR.value
    if target == "study":
        return Node.STUDY_CREATOR.value
    if target == "assay":
        return Node.ASSAY_CREATOR.value
    if target == "process":
        return Node.PROCESS_CREATOR.value
    if target == "protocol":
        return Node.PROTOCOL_CREATOR.value
    if target == "factor":
        return Node.FACTOR_CREATOR.value
    if target == "source":
        return Node.SOURCE_CREATOR.value
    if target == "sample":
        return Node.SAMPLE_CREATOR.value
    if target == "material":
        return Node.MATERIAL_CREATOR.value
    if target == "datafile":
        return Node.DATAFILE_CREATOR.value
    return Node.PLANNER_AGENT.value