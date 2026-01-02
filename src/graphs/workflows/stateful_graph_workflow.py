from src.graphs.nodes.ISA_updater import create_ISA_updater
from src.graphs.nodes.instructor import instructor
from src.model import get_chat_model
from src.graphs.nodes.planner import create_planner
from src.tools import create_doc_getter, create_schema_getter, validate_ISA_schema
from src.graphs.utils import index_document, stream_messages, NextStep, Action
from src.graphs.nodes.routing_function import routing_function
from langgraph.graph import StateGraph, START, END
from src.state.v2 import ISAExtractionStateV2
from src.graphs.nodes.tool_callers import read_schema, create_search_document_caller, validate
from langchain_core.runnables import RunnableConfig


def create_workflow_dependencies(): 
    """Creates dependencies required for the stateful graph workflow"""
    # Create doc manager and tools
    doc_manager = index_document()
    search_document_tool = create_doc_getter(doc_manager)
    search_document_caller = create_search_document_caller(search_document_tool)
    schema_getter_tools = create_schema_getter()

    return search_document_caller, schema_getter_tools


def create_stateful_graph_workflow() -> StateGraph:
    # Create workflow dependencies
    search_document_caller, schema_getter_tools = create_workflow_dependencies()

    # Create model for planner with structured output
    model_planner = get_chat_model().with_structured_output(NextStep)
   
    # Create planner node
    planner = create_planner(model_planner)

    # Create model for ISA updater
    model_updater = get_chat_model()
    model_updater.bind_tools(schema_getter_tools)

    isa_updater = create_ISA_updater(model_updater)

    graph = StateGraph(ISAExtractionStateV2)

    graph.add_node(Action.INSTRUCTOR.value, instructor)
    graph.add_node(Action.PLANNER.value, planner)
    graph.add_node(Action.UPDATE_DRAFT.value, isa_updater)
    graph.add_node(Action.SEARCH_DOCUMENT.value, search_document_caller)
    graph.add_node(Action.READ_SCHEMA.value, read_schema)
    graph.add_node(Action.VALIDATE.value, validate)

    graph.add_edge(START, Action.INSTRUCTOR.value)
    graph.add_edge(Action.INSTRUCTOR.value, Action.PLANNER.value)
    
    # Conditional edges from planner based on its decision
    graph.add_conditional_edges(
        Action.PLANNER.value, 
        routing_function,
        {
            Action.UPDATE_DRAFT.value: Action.UPDATE_DRAFT.value,
            Action.SEARCH_DOCUMENT.value: Action.SEARCH_DOCUMENT.value,
            Action.READ_SCHEMA.value: Action.READ_SCHEMA.value,
            Action.VALIDATE.value: Action.VALIDATE.value,
            Action.STOP.value: END,  # Allow planner to decide when to END
        }
    )

    # All action nodes route back to planner
    graph.add_edge(Action.UPDATE_DRAFT.value, Action.PLANNER.value)
    graph.add_edge(Action.SEARCH_DOCUMENT.value, Action.PLANNER.value)
    graph.add_edge(Action.READ_SCHEMA.value, Action.PLANNER.value)
    graph.add_edge(Action.VALIDATE.value, Action.PLANNER.value)

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