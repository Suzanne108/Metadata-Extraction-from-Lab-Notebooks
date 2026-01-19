import operator

from typing import TypedDict, Annotated, List

from langchain.messages import AnyMessage
from langchain_core.documents import Document
from src.graphs.types.NextStep import NextStep

from src.task import Task


class ISAExtractionState(TypedDict):
    """Extended state for staged workflow: overview -> decomposition -> planner loop."""

    # Document to extract from
    document: Document | List[Document]

    # Current ISA work in progress
    draft_isa_metadata: dict | None

    # The latest validation results
    validation_result: dict | None

    # Count of consecutive validation failures and last failing path (for guardrails)
    consecutive_validation_failures: int | None
    last_validation_json_path: str | None

    # The latest decision from the orchestrator
    orchestrator_decision: NextStep | None

    # Evidence per query
    evidence: dict[str, list[str]]

    # The latest retrieved evidence
    latest_evidence: list[str] | None

    # Orientation: plain-text ISA mental model
    isa_mental_model: str | None

    # Overview: plain-text structured overview with numbered entity IDs
    document_overview: str | None

    # Backlog of tasks produced by the task decomposer
    tasks: List[Task] | None

    # Index of the task currently being worked on
    current_task_index: int | None

    # Message tracing
    messages: Annotated[list[AnyMessage], operator.add]

    # The latest executed node/action in the planner loop
    # Values: "SEARCH_DOCUMENT" | "UPDATE_DRAFT" | "VALIDATE" | "TASK_PROGRESS" | None
    last_action: str | None
