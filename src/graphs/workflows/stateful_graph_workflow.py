from src.graphs.nodes.ISA_updater import create_ISA_updater
from src.model import get_chat_model
from src.graphs.nodes.planner import create_planner
from src.tools import create_doc_getter, create_schema_getter
from src.graphs.utils import index_document, stream_messages, NextStep, Node
from src.graphs.nodes.routing_function import routing_function
from langgraph.graph import StateGraph, START, END
from src.state.v2 import ISAExtractionStateV2
from src.graphs.nodes.tool_callers import create_search_document_caller, validate
from langchain_core.runnables import RunnableConfig
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.nodes.schema_summarizer import create_schema_summarizer


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
        registry.get_prompt("planner", "v3-system"), {})
    planner_policy_prompt = render_prompt(registry.get_prompt("planner", "v1-policy"), {})

    # Create planner node
    planner_node = create_planner(model_planner, planner_system_prompt, planner_policy_prompt)

    # Create model for ISA updater
    model_updater = get_chat_model().bind_tools(schema_getter_tools)

    updater_system_prompt = render_prompt(
        registry.get_prompt("ISA_updater", "v2-system"), {})
    
    updater_inference_prompt_template = registry.get_prompt("ISA_updater", "v2-inference")

    # Create ISA updater node
    isa_updater = create_ISA_updater(model_updater, schema_getter_tools, updater_system_prompt, updater_inference_prompt_template)

    # Create model for schema agent
    model_schema_agent = get_chat_model().bind_tools(schema_getter_tools)

    schema_agent_system_prompt = render_prompt(
        registry.get_prompt("schema", "v1-system"), {})
    
    schema_agent_inference_prompt_template = registry.get_prompt("schema", "v1-inference")
    
    # Create schema summarizer node
    schema_reader = create_schema_summarizer(model_schema_agent, schema_getter_tools, schema_agent_system_prompt, schema_agent_inference_prompt_template)


    graph = StateGraph(ISAExtractionStateV2)

    graph.add_node(Node.PLANNER_AGENT.value, planner_node)
    graph.add_node(Node.UPDATE_DRAFT_AGENT.value, isa_updater)
    graph.add_node(Node.SEARCH_DOCUMENT.value, search_document_caller)
    graph.add_node(Node.SCHEMA_SUMMARIZER_AGENT.value, schema_reader)
    graph.add_node(Node.VALIDATE.value, validate)

    graph.add_edge(START, Node.PLANNER_AGENT.value)
    
    # Conditional edges from planner based on its decision
    graph.add_conditional_edges(
        Node.PLANNER_AGENT.value, 
        routing_function,
        {
            Node.UPDATE_DRAFT_AGENT.value: Node.UPDATE_DRAFT_AGENT.value,
            Node.SEARCH_DOCUMENT.value: Node.SEARCH_DOCUMENT.value,
            Node.SCHEMA_SUMMARIZER_AGENT.value: Node.SCHEMA_SUMMARIZER_AGENT.value,
            Node.VALIDATE.value: Node.VALIDATE.value,
            Node.STOP.value: END,  # Allow planner to decide when to END
        }
    )

    # All action nodes route back to planner
    graph.add_edge(Node.UPDATE_DRAFT_AGENT.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.SEARCH_DOCUMENT.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.SCHEMA_SUMMARIZER_AGENT.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.VALIDATE.value, Node.PLANNER_AGENT.value)

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