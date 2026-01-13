from typing import TypedDict, Annotated, Any, List, Literal
from langchain.messages import AnyMessage
import operator
from langchain_core.documents import Document
from src.graphs.utils import NextStep
from pydantic import BaseModel


class Task(BaseModel):
    kind: Literal["investigation", "study", "assay", "material", "process", "protocol", "factor", "data"]
    target: str
    instruction: str
    status: Literal["pending", "in-progress", "completed"]


class TaskList(BaseModel):
    tasks: List[Task]


class ISAExtractionStateV3(TypedDict):
    """Extended state for staged workflow: overview -> decomposition -> planner loop."""

    # Document to extract from
    document: Document | List[Document]

    # Current ISA work in progress
    draft_isa_metadata: dict | None

    # The latest validation results
    validation_result: dict | None

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
