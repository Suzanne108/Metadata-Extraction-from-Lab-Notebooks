from src.graphs.utils import write_final_state
from src.state import ISAExtractionState

#TODO: create a separate folder for specialist-agent workflow, where all its relevant code is located.


def handle_final_state(app, config, run_id) -> ISAExtractionState | None:
    """Retrieve the final state snapshot from the compiled graph and persist it."""
    try:
        state_snapshot = app.get_state(config)
        write_final_state(state_snapshot, run_id)
        return state_snapshot
    except Exception as e:
        print(f"Failed to retrieve final state: {e}")
        return None
