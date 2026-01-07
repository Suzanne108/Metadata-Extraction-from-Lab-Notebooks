from src.state.v2 import ISAExtractionStateV2


def routing_function(state: ISAExtractionStateV2) -> str:
    return state['orchestrator_decision'].action.value