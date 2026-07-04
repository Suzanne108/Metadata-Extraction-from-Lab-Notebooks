from src.state import ISAExtractionState
from src.graphs.types.Action import Action
from src.graphs.types.Node import GraphComponentId

#TODO: give the routing functions a more descriptive name, e.g. "planner_routing_function" and "team_lead_routing_function" to avoid confusion with the generic "routing_function" in the graph builder

#TODO: move this file to another folder
def routing_function(state: ISAExtractionState) -> str:
    """Map the planner's enum action to a concrete graph node label.

    Using explicit enum mapping avoids brittle string matching that can
    silently fail to route when labels change or differ in whitespace/casing.
    """
    decision = state.get("planner_decision")
    if decision is None or getattr(decision, "action", None) is None:
        # Fallback to planner loop if we somehow have no decision
        return GraphComponentId.PLANNER_AGENT.value

    action: Action = decision.action

    if action == Action.UPDATE_DRAFT:
        # Route to team lead for specialized creation routing
        return GraphComponentId.TEAM_LEAD.value
    if action == Action.SEARCH_DOCUMENT:
        return GraphComponentId.SEARCH_DOCUMENT.value
    if action == Action.FINISH_TASK:
        return GraphComponentId.TASK_PROGRESS.value
    if action == Action.STOP:
        # When planner signals completion, first run assembly advice before assembly/export
        return GraphComponentId.ASSEMBLY_ADVISOR.value

    # Default: keep planner in control
    return GraphComponentId.PLANNER_AGENT.value


def team_lead_routing(state: ISAExtractionState) -> str:
    """Route based on team lead decision stored in state."""
    decision = state.get("team_lead_decision")
    target = getattr(decision, "target", None)
    if target == "investigation":
        return GraphComponentId.INVESTIGATION_CREATOR.value
    if target == "study":
        return GraphComponentId.STUDY_CREATOR.value
    if target == "assay":
        return GraphComponentId.ASSAY_CREATOR.value
    if target == "process":
        return GraphComponentId.PROCESS_CREATOR.value
    if target == "protocol":
        return GraphComponentId.PROTOCOL_CREATOR.value
    if target == "factor":
        return GraphComponentId.FACTOR_CREATOR.value
    if target == "source":
        return GraphComponentId.SOURCE_CREATOR.value
    if target == "sample":
        return GraphComponentId.SAMPLE_CREATOR.value
    if target == "material":
        return GraphComponentId.MATERIAL_CREATOR.value
    if target == "datafile":
        return GraphComponentId.DATAFILE_CREATOR.value
    return GraphComponentId.PLANNER_AGENT.value