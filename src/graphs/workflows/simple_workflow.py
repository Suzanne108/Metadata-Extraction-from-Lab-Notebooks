from pathlib import Path

from src.agents.simple_agent import get_simple_agent_v1
from langchain.messages import HumanMessage
from src.graphs.utils import index_document




def simple_workflow_v1():
    doc_manager = index_document()
    agent = get_simple_agent_v1(doc_manager)

    question = "Please extract the title and authors from the given document."

    state = {
        "messages": [HumanMessage(content = question)],
        "document": doc_manager.document
    }

    result = agent.invoke(state)
    for message in result["messages"]:
        message.pretty_print()