from src.graphs.nodes.ISA_updater import create_ISA_updater
from src.graphs.nodes.isa_overview import create_isa_overview
from src.graphs.nodes.document_overview import create_document_overview
from src.graphs.nodes.task_decomposer import create_task_decomposer
from src.model import get_chat_model
from src.graphs.nodes.planner import create_planner
from src.tools import create_doc_getter, create_schema_getter
from src.graphs.utils import index_document, stream_messages, NextStep, Node
from src.graphs.nodes.routing_function import routing_function
from langgraph.graph import StateGraph, START, END
from src.state.v3 import ISAExtractionStateV3, TaskList
from src.graphs.nodes.tool_callers import create_search_document_caller, validate
from langchain_core.runnables import RunnableConfig
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.nodes.schema_summarizer import create_schema_summarizer
from src.graphs.nodes.task_progress import create_task_progressor


def create_workflow_dependencies():
    """Creates dependencies required for the stateful graph workflow"""
    # Create doc manager and tools
    doc_manager = index_document()
    search_document_tool = create_doc_getter(doc_manager)
    search_document_caller = create_search_document_caller(search_document_tool)
    schema_getter_tools = create_schema_getter()

    return search_document_caller, schema_getter_tools


def build_planner_node(registry: PromptRegistry):
    """Create the planner node with structured output and prompts."""
    model_planner = get_chat_model().with_structured_output(NextStep)
    planner_system_prompt = render_prompt(registry.get_prompt("planner", "v4-system"), {})
    planner_policy_prompt = render_prompt(registry.get_prompt("planner", "v2-policy"), {})
    return create_planner(model_planner, planner_system_prompt, planner_policy_prompt)

def build_isa_overview_node(registry: PromptRegistry, schema_getter_tools):
    """Create the ISA overview node with its prompts."""

    model = get_chat_model().bind_tools(schema_getter_tools)
    isa_overview_system = render_prompt(registry.get_prompt("ISA_overview", "v1-system"), {})
    isa_overview_infer = registry.get_prompt("ISA_overview", "v1-inference")

    return create_isa_overview(model, schema_getter_tools, isa_overview_system, isa_overview_infer)


def build_doc_overview_node(registry: PromptRegistry):
    """Create ISA overview and document overview nodes with their prompts."""
    model_llm = get_chat_model()

    # Document Overview
    doc_overview_system = render_prompt(registry.get_prompt("document_overview", "v1-system"), {})
    doc_overview_infer = registry.get_prompt("document_overview", "v1-inference")
    doc_overview_node = create_document_overview(model_llm, doc_overview_system, doc_overview_infer)

    return doc_overview_node


def build_task_decomposer_node(registry: PromptRegistry):
    """Create the task decomposer node using a dedicated structured-output model."""
    task_decomp_system = render_prompt(registry.get_prompt("task_decomposer", "v1-system"), {})
    task_decomp_infer = registry.get_prompt("task_decomposer", "v1-inference")
    model_task_decomposer = get_chat_model().with_structured_output(TaskList)
    return create_task_decomposer(model_task_decomposer, task_decomp_system, task_decomp_infer)


def build_schema_agent_node(registry: PromptRegistry, schema_getter_tools):
    """Create schema summarizer node bound with schema tools."""
    model_schema_agent = get_chat_model().bind_tools(schema_getter_tools)
    schema_agent_system_prompt = render_prompt(registry.get_prompt("schema", "v1-system"), {})
    schema_agent_inference_prompt_template = registry.get_prompt("schema", "v1-inference")
    return create_schema_summarizer(model_schema_agent, schema_getter_tools, schema_agent_system_prompt, schema_agent_inference_prompt_template)


def build_isa_updater_node(registry: PromptRegistry, schema_getter_tools):
    """Create ISA updater node bound with schema tools."""
    model_updater = get_chat_model().bind_tools(schema_getter_tools)
    updater_system_prompt = render_prompt(registry.get_prompt("ISA_updater", "v3-system"), {})
    updater_inference_prompt_template = registry.get_prompt("ISA_updater", "v2-inference")
    return create_ISA_updater(model_updater, schema_getter_tools, updater_system_prompt, updater_inference_prompt_template)


def build_graph(
    isa_overview_node,
    doc_overview_node,
    task_decomposer_node,
    task_progress_node,
    planner_node,
    search_document_caller,
   # schema_reader,
    isa_updater,
):
    """Add nodes in order of appearance and wire edges."""
    graph = StateGraph(ISAExtractionStateV3)

    # Create nodes in the order they appear in the graph
    graph.add_node(Node.ISA_OVERVIEW.value, isa_overview_node)
    graph.add_node(Node.DOC_OVERVIEW.value, doc_overview_node)
    graph.add_node(Node.TASK_DECOMPOSER.value, task_decomposer_node)
    graph.add_node(Node.TASK_PROGRESS.value, task_progress_node)
    graph.add_node(Node.PLANNER_AGENT.value, planner_node)
    graph.add_node(Node.SEARCH_DOCUMENT.value, search_document_caller)
   # graph.add_node(Node.SCHEMA_SUMMARIZER_AGENT.value, schema_reader)
    graph.add_node(Node.UPDATE_DRAFT_AGENT.value, isa_updater)
    graph.add_node(Node.VALIDATE.value, validate)

    # Linear pre-loop sequence
    graph.add_edge(START, Node.ISA_OVERVIEW.value)
    graph.add_edge(Node.ISA_OVERVIEW.value, Node.DOC_OVERVIEW.value)
    graph.add_edge(Node.DOC_OVERVIEW.value, Node.TASK_DECOMPOSER.value)
    graph.add_edge(Node.TASK_DECOMPOSER.value, Node.TASK_PROGRESS.value)
    graph.add_edge(Node.TASK_PROGRESS.value, Node.PLANNER_AGENT.value)

    # Conditional edges from planner based on its decision
    graph.add_conditional_edges(
        Node.PLANNER_AGENT.value,
        routing_function,
        {
            Node.UPDATE_DRAFT_AGENT.value: Node.UPDATE_DRAFT_AGENT.value,
            Node.SEARCH_DOCUMENT.value: Node.SEARCH_DOCUMENT.value,
            Node.TASK_PROGRESS.value: Node.TASK_PROGRESS.value,
           # Node.SCHEMA_SUMMARIZER_AGENT.value: Node.SCHEMA_SUMMARIZER_AGENT.value,
            Node.VALIDATE.value: Node.VALIDATE.value,
            Node.STOP.value: END,  # Allow planner to decide when to END
        },
    )

    # All action nodes route back to planner
    graph.add_edge(Node.UPDATE_DRAFT_AGENT.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.SEARCH_DOCUMENT.value, Node.PLANNER_AGENT.value)
    #graph.add_edge(Node.SCHEMA_SUMMARIZER_AGENT.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.VALIDATE.value, Node.PLANNER_AGENT.value)

    return graph.compile()


def create_stateful_graph_workflow() -> StateGraph:
    # Dependencies
    search_document_caller, schema_getter_tools = create_workflow_dependencies()

    # Registry
    registry = PromptRegistry()

    # Nodes in order of appearance
    isa_overview_node = build_isa_overview_node(registry, schema_getter_tools)
    doc_overview_node = build_doc_overview_node(registry)
    task_decomposer_node = build_task_decomposer_node(registry)
    planner_node = build_planner_node(registry)
    task_progress_node = create_task_progressor()
    # schema_reader = build_schema_agent_node(registry, schema_getter_tools)
    isa_updater = build_isa_updater_node(registry, schema_getter_tools)

    # Build and compile graph
    app = build_graph(
        isa_overview_node,
        doc_overview_node,
        task_decomposer_node,
        task_progress_node,
        planner_node,
        search_document_caller,
        isa_updater,
    )

    return app

def run_stateful_graph_workflow():
    app = create_stateful_graph_workflow()

    doc_manager = index_document()
    document = doc_manager.document

    # Initialize empty state
    initial_state = ISAExtractionStateV3(
        document = document,
        messages=[],
        draft_isa_metadata={},
        evidence={},
        validation_result={},
        isa_mental_model="",
        document_overview="",
        tasks=[]
    )

    config = RunnableConfig(recursion_limit=50)

    stream_messages(app, initial_state, config=config)