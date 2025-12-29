
from src.agents.standalone_extraction_agent import get_standalone_extraction_agent
from langchain.messages import HumanMessage
from src.graphs.utils import index_document, stream


def standalone_agent_workflow():
    doc_manager = index_document()
    agent = get_standalone_extraction_agent(doc_manager)

    question = "Create ISA metadata for the provided document"

    state = {
        "messages": [HumanMessage(content = question)],
        "document": doc_manager.document
    }

    stream(agent, state)