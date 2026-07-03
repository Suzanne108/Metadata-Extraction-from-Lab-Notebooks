import uuid

from langchain_core.runnables import RunnableConfig

from src.graphs.utils import stream_messages
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import create_stateful_graph_workflow
from src.graphs.workflows.isa_extraction_workflow.final_state import handle_final_state
from src.state import ISAExtractionState

#TODO: make this method general, so that any kind of workflow can be run using this method.


def run_stateful_graph_workflow(
    document_path: str,
    initial_state: ISAExtractionState,
) -> ISAExtractionState | None:
    """Build the graph, stream the workflow, and return the final state.

    Args:
        document_path: Path to the document to extract metadata from.
        initial_state: Pre-built initial state (created by ``create_initial_state``).
    """
    app = create_stateful_graph_workflow(document_path)

    run_id = uuid.uuid4()
    thread_id = f"isa-extract-{run_id}"
    config = RunnableConfig(
        recursion_limit=9000000,
        configurable={"thread_id": thread_id},
    )

    stream_messages(run_id, app, initial_state, config=config)

    return handle_final_state(app, config, run_id)
