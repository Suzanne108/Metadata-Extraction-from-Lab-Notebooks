from langchain.tools import tool
from src.document_processing import DocumentManager
from langchain_core.documents import Document
from typing import List, Dict

def create_doc_getter(doc_manager: DocumentManager) -> List:
    """Creates tools with access to their dependencies"""

    @tool
    def get_doc_chunk(query: str, k: int = 4) -> List[Document]:
        """ Retrieves relevant chunks from the extraction document """

        # The doc manager must be retrieved from the context
        chunks = doc_manager.retrieve_semantic_chunk(query, k)

        return chunks

    return get_doc_chunk

@tool
def get_ISA_schema() -> Dict:
    """ Fetches the specified ISA schema """
    #TODO: implement this
    return {}

@tool
def validate_ISA_schema() -> Dict:
    """ Validates the given ISA schema """
    #TODO: implement this
    return {}

    