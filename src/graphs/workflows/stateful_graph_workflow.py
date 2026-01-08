from src.graphs.nodes.ISA_updater import create_ISA_updater
from src.model import get_chat_model
from src.graphs.nodes.planner import create_planner
from src.tools import create_doc_getter, create_schema_getter
from src.graphs.utils import index_document, stream_messages, NextStep, Action, DraftUpdate, Node
from src.graphs.nodes.routing_function import routing_function
from langgraph.graph import StateGraph, START, END
from src.state.v2 import ISAExtractionStateV2
from src.graphs.nodes.tool_callers import read_schema, create_search_document_caller, validate
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from langchain.messages import SystemMessage


def create_workflow_dependencies(): 
    """Creates dependencies required for the stateful graph workflow"""
    # Create doc manager and tools
    doc_manager = index_document()
    search_document_tool = create_doc_getter(doc_manager)
    search_document_caller = create_search_document_caller(search_document_tool)
    schema_getter_tools = create_schema_getter()

    return search_document_caller, schema_getter_tools


def create_stateful_graph_workflow() -> StateGraph:
    #TODO: refactor to separate functions

    # Create workflow dependencies
    search_document_caller, schema_getter_tools = create_workflow_dependencies()

    # Create model for planner with structured output
    model_planner = get_chat_model().with_structured_output(NextStep)

    registry = PromptRegistry()
    planner_system_prompt = render_prompt(
        registry.get_prompt("planner", "v2-system"), {})

    # Create planner node
    planner_node = create_planner(model_planner, planner_system_prompt)

    # Create model for ISA updater
    model_updater = get_chat_model().bind_tools(schema_getter_tools).with_structured_output(DraftUpdate)

    updater_system_prompt = render_prompt(
        registry.get_prompt("ISA_updater", "v2-system"), {})
    
    updater_inference_prompt_template = registry.get_prompt("ISA_updater", "v2-inference")

    # Create ISA updater node
    isa_updater = create_ISA_updater(model_updater, updater_system_prompt, updater_inference_prompt_template)

    graph = StateGraph(ISAExtractionStateV2)

    graph.add_node(Node.PLANNER.value, planner_node)
    graph.add_node(Node.UPDATE_DRAFT.value, isa_updater)
    graph.add_node(Node.SEARCH_DOCUMENT.value, search_document_caller)
    graph.add_node(Node.READ_SCHEMA.value, read_schema)
    graph.add_node(Node.VALIDATE.value, validate)

    graph.add_edge(START, Node.PLANNER.value)
    
    # Conditional edges from planner based on its decision
    graph.add_conditional_edges(
        Node.PLANNER.value, 
        routing_function,
        {
            Node.UPDATE_DRAFT.value: Node.UPDATE_DRAFT.value,
            Node.SEARCH_DOCUMENT.value: Node.SEARCH_DOCUMENT.value,
            Node.READ_SCHEMA.value: Node.READ_SCHEMA.value,
            Node.VALIDATE.value: Node.VALIDATE.value,
            Node.STOP.value: END,  # Allow planner to decide when to END
        }
    )

    # All action nodes route back to planner
    graph.add_edge(Node.UPDATE_DRAFT.value, Node.PLANNER.value)
    graph.add_edge(Node.SEARCH_DOCUMENT.value, Node.PLANNER.value)
    graph.add_edge(Node.READ_SCHEMA.value, Node.PLANNER.value)
    graph.add_edge(Node.VALIDATE.value, Node.PLANNER.value)

    app = graph.compile()

    return app

def run_stateful_graph_workflow():
    app = create_stateful_graph_workflow()

    doc_manager = index_document()
    document = doc_manager.document

    # Initialize empty state
    initial_state = ISAExtractionStateV2(
        document = document,
        messages=[],
        draft_isa_metadata={},
        schema_cache={},
        evidence={},
        validation_result={}
    )

    config = RunnableConfig(recursion_limit=50)

    stream_messages(app, initial_state, config=config)