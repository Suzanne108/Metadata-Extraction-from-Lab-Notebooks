from typing import TypedDict, Annotated, Any
from langchain.messages import AnyMessage
import operator
from langchain_core.documents import Document
from src.graphs.utils import NextStep


class ISAExtractionStateV2(TypedDict):
    """ This class defines a simple state passed between nodes of the workflow """

    # Document to extract from
    document: Document

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

    # The amount of iterations (not sure if should be included now)
    #iteration: int
    #max_iterations: int


    # Message tracing
    messages: Annotated[list[AnyMessage], operator.add] 
