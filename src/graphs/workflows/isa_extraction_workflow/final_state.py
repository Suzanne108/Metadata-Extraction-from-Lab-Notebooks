from src.graphs.utils import write_final_state
from src.state import ISAExtractionState

# Move the handle final state functionality from utils to here


def handle_final_state(app, config, run_id) -> ISAExtractionState | None:
    """Retrieve the final state snapshot from the compiled graph and persist it."""
    try:
        state_snapshot = app.get_state(config)
        write_final_state(state_snapshot, run_id)
        return state_snapshot
    except Exception as e:
        print(f"Failed to retrieve final state: {e}")
        return None
