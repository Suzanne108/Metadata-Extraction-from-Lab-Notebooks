from src.document_processing import DocumentManager
from src.state import ISAExtractionState

def create_initial_state(document_manager: DocumentManager) -> ISAExtractionState:
    """Return the initial workflow state from a preloaded document manager."""
    if document_manager.document is None:
        raise ValueError("document_manager must load a document before creating the initial state.")

    document = document_manager.document

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
