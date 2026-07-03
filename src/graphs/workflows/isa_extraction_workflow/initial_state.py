from src.graphs.utils import index_document
from src.state import ISAExtractionState

#TODO: refactor initial state and final state into some kind of larger class that can also run and build the workflow


def create_initial_state(document_path: str) -> ISAExtractionState:
    """Index the document and return the initial workflow state."""
    doc_manager = index_document(document_path)
    document = doc_manager.document

    return ISAExtractionState(
        document=document,
        messages=[],
        draft_isa_metadata={},
        evidence={},
        validation_result={},
        isa_mental_model="",
        document_overview="",
        tasks=[],
        last_action=None,
    )
