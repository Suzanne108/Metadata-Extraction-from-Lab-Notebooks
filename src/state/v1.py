from typing import TypedDict, Annotated
from langchain.messages import AnyMessage
import operator
from langchain_core.documents import Document


class ISAExtractionStateV1(TypedDict):
    """ This class defines a simple state passed between nodes of the workflow """

    # Document to extract from
    document: Document

    # Generated content
    draft_isa_metadata: dict | None
    messages: Annotated[list[AnyMessage], operator.add] 
