from pathlib import Path

from src.agents.simple_agent import get_simple_agent
from src.document_processing import DocumentManager
from langchain.messages import HumanMessage
from langchain_core.documents import Document

def index_document() -> DocumentManager:
    path = Path.cwd() / "papers" / "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf "
    doc_manager = DocumentManager()

    doc_manager.embed_documents(path)

    return doc_manager


def simple_workflow():
    doc_manager = index_document()
    agent = get_simple_agent(doc_manager)

    question = "Please extract the title and authors from the given document."

    state = {
        "messages": [HumanMessage(content = question)],
        "document": doc_manager.document
    }

    result = agent.invoke(state)
    for message in result["messages"]:
        message.pretty_print()