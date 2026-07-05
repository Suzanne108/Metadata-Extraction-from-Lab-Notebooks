import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.graphs.utils import stream_messages


def run_graph_workflow(
    app: Any,
    initial_state: Any,
    run_prefix: str = "workflow",
    stream: bool = True,
) -> tuple[Any, RunnableConfig, uuid.UUID]:
    """Run a compiled workflow with optional streaming and return run context.

    Args:
        app: Compiled workflow application with ``stream``/``invoke`` support.
        initial_state: Initial state passed to the workflow.
        run_prefix: Prefix used in the generated thread identifier.
        stream: Whether to stream intermediate node updates.
    """
    run_id = uuid.uuid4()
    thread_id = f"{run_prefix}-{run_id}"
    config = RunnableConfig(
        recursion_limit=9000000,
        configurable={"thread_id": thread_id},
    )

    if stream:
        stream_messages(run_id, app, initial_state, config=config)
    else:
        app.invoke(initial_state, config=config)

    return app, config, run_id
