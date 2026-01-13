from src.state.v3 import ISAExtractionStateV3


def routing_function(state: ISAExtractionStateV3) -> str:
    return state['orchestrator_decision'].action.value