""" This file defines State classes which can be passed between nodes of a workflow """

from typing import TypedDict, Annotated
from langchain.messages import AnyMessage
import operator
from langchain_core.documents import Document


class ISAExtractionState(TypedDict):
    """ This class defines a simple state passed between nodes of the workflow """

    # Document to extract from
    document: Document

    # Generated content
    draft_isa_metadata: dict | None
    messages: Annotated[list[AnyMessage], operator.add] 

    #TODO: hard to define what to include in state. Things to consider:
    # - the raw document to extract from
    # - the retrieved document chunks
    # - the ISA schema's? 
