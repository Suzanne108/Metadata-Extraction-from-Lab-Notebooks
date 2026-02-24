import uuid
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from src.graphs.nodes.ISA_updater import create_ISA_updater
from src.graphs.nodes.isa_overview import create_isa_overview
from src.graphs.nodes.document_overview import create_document_overview
from src.graphs.nodes.task_decomposer import create_task_decomposer
from src.model import get_chat_model
from src.graphs.nodes.planner import create_planner
from src.tools import create_doc_getter, create_schema_getter
from src.graphs.utils import index_document, stream_messages, write_final_state
from src.graphs.nodes.routing_function import routing_function, team_lead_routing
from src.graphs.nodes.assembly import create_assembly_node
from src.graphs.nodes.assembly_advisor import create_assembly_advisor
from src.graphs.nodes.investigation_export import create_investigation_export_node
from src.state import ISAExtractionState
from src.graphs.nodes.tool_callers import create_search_document_caller, validate
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.nodes.task_progress import create_task_progressor
from src.document_processing import DocumentManager
from src.graphs.types.NextStep import NextStepLLM
from src.graphs.types.Node import Node
from src.task import TaskList
from src.graphs.nodes.isa_team_lead import create_isa_team_lead
from src.graphs.nodes.specialist_agent import create_specialist_agent
from src.graphs.types.TeamLeadDecision import TeamLeadDecision
from src.graphs.types.DTOs.InvestigationDTO import InvestigationDTO
from src.graphs.types.DTOs.StudyDTO import StudyDTO
from src.graphs.types.DTOs.AssayDTO import AssayDTO
from src.graphs.types.DTOs.ProcessDTO import ProcessDTO
from src.graphs.types.DTOs.ProtocolDTO import ProtocolDTO
from src.graphs.types.DTOs.FactorDTO import FactorDTO
from src.graphs.types.DTOs.SourceDTO import SourceDTO
from src.graphs.types.DTOs.SampleDTO import SampleDTO
from src.graphs.types.DTOs.MaterialDTO import MaterialDTO
from src.graphs.types.DTOs.DataDTO import DataFileDTO



def create_workflow_dependencies(document_path: str = None):
    """Creates dependencies required for the stateful graph workflow"""
    # Create doc manager and tools
    doc_manager = index_document(document_path=document_path)
    search_document_tool = create_doc_getter(doc_manager)
    search_document_caller = create_search_document_caller(search_document_tool)
    schema_getter_tools = create_schema_getter()

    return search_document_caller, schema_getter_tools


def build_planner_node(registry: PromptRegistry):
    """Create the planner node with structured output and prompts."""
    model_planner = get_chat_model().with_structured_output(NextStepLLM)
    planner_system_prompt = render_prompt(registry.get_prompt("planner", "v5-system"), {})
    planner_policy_prompt = render_prompt(registry.get_prompt("planner", "v3-policy"), {})
    return create_planner(model_planner, planner_system_prompt, planner_policy_prompt)

def build_isa_overview_node(registry: PromptRegistry, schema_getter_tools):
    """Create the ISA overview node with its prompts."""

    model = get_chat_model().bind_tools(schema_getter_tools)
    isa_overview_system = render_prompt(registry.get_prompt("ISA_overview", "v2-system"), {})
    isa_overview_infer = registry.get_prompt("ISA_overview", "v2-inference")

    doc_manager = DocumentManager()
    path = Path.cwd() / "instructions" / "ISA_abstract_model.pdf"

    isa_overview_doc = doc_manager.load_pdf(path)

    return create_isa_overview(model, schema_getter_tools, isa_overview_system, isa_overview_infer, isa_overview_doc)

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

def build_isa_updater_node(registry: PromptRegistry, schema_getter_tools):
    """Create ISA updater node bound with schema tools."""
    model_updater = get_chat_model().bind_tools(schema_getter_tools)
    updater_system_prompt = render_prompt(registry.get_prompt("ISA_updater", "v3-system"), {})
    updater_inference_prompt_template = registry.get_prompt("ISA_updater", "v2-inference")
    return create_ISA_updater(model_updater, schema_getter_tools, updater_system_prompt, updater_inference_prompt_template)


def build_team_lead_node(registry: PromptRegistry):
    """Create ISA creation team lead node with prompts."""
    model = get_chat_model().with_structured_output(TeamLeadDecision)
    system_prompt = render_prompt(registry.get_prompt("isa_team_lead", "v1-system"), {})
    infer_prompt = registry.get_prompt("isa_team_lead", "v1-inference")
    
    return create_isa_team_lead(model, system_prompt, infer_prompt)


def build_specialist_node(
    registry: PromptRegistry,
    dto_cls,
    prompt_key: str,
    dto_state_key: str,
    agent_label: str,
):
    """Generic builder for specialist agent nodes.

    Args:
        registry: Prompt registry instance
        dto_cls: Pydantic DTO class for structured output
        prompt_key: subfolder key in prompts (e.g., "assay_specialist")
        dto_state_key: state key to store the resulting DTO (e.g., "dto_assay")
        agent_label: human-friendly label for logging (e.g., "ASSAY_AGENT")
    """
    model = get_chat_model().with_structured_output(dto_cls)
    system_prompt = render_prompt(registry.get_prompt(prompt_key, "v1-system"), {})
    infer_prompt = registry.get_prompt(prompt_key, "v1-inference")
    return create_specialist_agent(model, system_prompt, infer_prompt, dto_state_key, agent_label)


def build_investigation_agent_node(registry: PromptRegistry):
    """Create investigation specialist agent node with prompts."""
    return build_specialist_node(
        registry,
        InvestigationDTO,
        "investigation_specialist",
        "dto_investigation",
        "INVESTIGATION_AGENT",
    )

def build_study_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        StudyDTO,
        "study_specialist",
        "dto_study",
        "STUDY_AGENT",
    )

def build_assay_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        AssayDTO,
        "assay_specialist",
        "dto_assay",
        "ASSAY_AGENT",
    )

def build_process_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        ProcessDTO,
        "process_specialist",
        "dto_process",
        "PROCESS_AGENT",
    )

def build_protocol_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        ProtocolDTO,
        "protocol_specialist",
        "dto_protocol",
        "PROTOCOL_AGENT",
    )

def build_factor_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        FactorDTO,
        "factor_specialist",
        "dto_factor",
        "FACTOR_AGENT",
    )

def build_source_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        SourceDTO,
        "source_specialist",
        "dto_source",
        "SOURCE_AGENT",
    )

def build_sample_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        SampleDTO,
        "sample_specialist",
        "dto_sample",
        "SAMPLE_AGENT",
    )

def build_material_agent_node(registry: PromptRegistry):
    return build_specialist_node(
        registry,
        MaterialDTO,
        "material_specialist",
        "dto_material",
        "MATERIAL_AGENT",
    )

def build_datafile_agent_node(registry: PromptRegistry):
    """Create datafile specialist agent node with prompts."""
    return build_specialist_node(
        registry,
        DataFileDTO,
        "datafile_specialist",
        "dto_data",
        "DATAFILE_AGENT",
    )

#TODO: Refactor this code so that there is not one method per node build, but a general node builder function

def build_export_node():
    """Build the simple export node that writes ISA-JSON."""
    return create_investigation_export_node()

def build_assembly_advisor_node(registry: PromptRegistry):
    """Create the assembly advisor node with its prompts."""
    model = get_chat_model()
    system_prompt = render_prompt(registry.get_prompt("assembly_advisor", "v1-system"), {})
    infer_prompt = registry.get_prompt("assembly_advisor", "v1-inference")
    return create_assembly_advisor(model, system_prompt, infer_prompt)


def build_graph(
    isa_overview_node,
    doc_overview_node,
    task_decomposer_node,
    task_progress_node,
    planner_node,
    search_document_caller,
    team_lead_node,
    investigation_agent_node,
    study_agent_node,
    assay_agent_node,
    process_agent_node,
    protocol_agent_node,
    factor_agent_node,
    source_agent_node,
    sample_agent_node,
    material_agent_node,
    datafile_agent_node,
):
    """Add nodes in order of appearance and wire edges."""
    graph = StateGraph(ISAExtractionState)

    # Create nodes in the order they appear in the graph
    graph.add_node(Node.ISA_OVERVIEW.value, isa_overview_node)
    graph.add_node(Node.DOC_OVERVIEW.value, doc_overview_node)
    graph.add_node(Node.TASK_DECOMPOSER.value, task_decomposer_node)
    graph.add_node(Node.TASK_PROGRESS.value, task_progress_node)
    graph.add_node(Node.PLANNER_AGENT.value, planner_node)
    graph.add_node(Node.SEARCH_DOCUMENT.value, search_document_caller)
    graph.add_node(Node.TEAM_LEAD.value, team_lead_node)
    graph.add_node(Node.INVESTIGATION_CREATOR.value, investigation_agent_node)
    graph.add_node(Node.STUDY_CREATOR.value, study_agent_node)
    graph.add_node(Node.ASSAY_CREATOR.value, assay_agent_node)
    graph.add_node(Node.PROCESS_CREATOR.value, process_agent_node)
    graph.add_node(Node.PROTOCOL_CREATOR.value, protocol_agent_node)
    graph.add_node(Node.FACTOR_CREATOR.value, factor_agent_node)
    graph.add_node(Node.SOURCE_CREATOR.value, source_agent_node)
    graph.add_node(Node.SAMPLE_CREATOR.value, sample_agent_node)
    graph.add_node(Node.MATERIAL_CREATOR.value, material_agent_node)
    graph.add_node(Node.DATAFILE_CREATOR.value, datafile_agent_node)
    #graph.add_node(Node.VALIDATE.value, validate)
    # Advisor, deterministic assembly, then export
    graph.add_node(Node.ASSEMBLY_ADVISOR.value, build_assembly_advisor_node(PromptRegistry()))
    graph.add_node(Node.ASSEMBLY.value, create_assembly_node())
    graph.add_node(Node.EXPORT.value, build_export_node())

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
            Node.TEAM_LEAD.value: Node.TEAM_LEAD.value,
            Node.SEARCH_DOCUMENT.value: Node.SEARCH_DOCUMENT.value,
            Node.TASK_PROGRESS.value: Node.TASK_PROGRESS.value,
            #Node.VALIDATE.value: Node.VALIDATE.value,
            Node.ASSEMBLY_ADVISOR.value: Node.ASSEMBLY_ADVISOR.value,
        },
    )

    # From team lead, route to specialized creation agents
    graph.add_conditional_edges(
        Node.TEAM_LEAD.value,
        team_lead_routing,
        {
            Node.INVESTIGATION_CREATOR.value: Node.INVESTIGATION_CREATOR.value,
            Node.STUDY_CREATOR.value: Node.STUDY_CREATOR.value,
            Node.ASSAY_CREATOR.value: Node.ASSAY_CREATOR.value,
            Node.PROCESS_CREATOR.value: Node.PROCESS_CREATOR.value,
            Node.PROTOCOL_CREATOR.value: Node.PROTOCOL_CREATOR.value,
            Node.FACTOR_CREATOR.value: Node.FACTOR_CREATOR.value,
            Node.SOURCE_CREATOR.value: Node.SOURCE_CREATOR.value,
            Node.SAMPLE_CREATOR.value: Node.SAMPLE_CREATOR.value,
            Node.MATERIAL_CREATOR.value: Node.MATERIAL_CREATOR.value,
            Node.DATAFILE_CREATOR.value: Node.DATAFILE_CREATOR.value,
            Node.PLANNER_AGENT.value: Node.PLANNER_AGENT.value,
        },
    )

    # All action nodes route back to planner
    graph.add_edge(Node.INVESTIGATION_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.STUDY_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.ASSAY_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.PROCESS_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.PROTOCOL_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.FACTOR_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.SOURCE_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.SAMPLE_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.MATERIAL_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.DATAFILE_CREATOR.value, Node.PLANNER_AGENT.value)
    graph.add_edge(Node.SEARCH_DOCUMENT.value, Node.PLANNER_AGENT.value)
    #graph.add_edge(Node.VALIDATE.value, Node.PLANNER_AGENT.value)
    # After advisor, run deterministic assembly, then export, then end
    graph.add_edge(Node.ASSEMBLY_ADVISOR.value, Node.ASSEMBLY.value)
    graph.add_edge(Node.ASSEMBLY.value, Node.EXPORT.value)
    graph.add_edge(Node.EXPORT.value, END)

    # Compile with an in-memory checkpointer so state can be retrieved
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def create_stateful_graph_workflow(document_path: str = None) -> StateGraph:
    # Dependencies
    search_document_caller, schema_getter_tools = create_workflow_dependencies(document_path)

    # Registry
    registry = PromptRegistry()

    # Nodes in order of appearance
    isa_overview_node = build_isa_overview_node(registry, schema_getter_tools)
    doc_overview_node = build_doc_overview_node(registry)
    task_decomposer_node = build_task_decomposer_node(registry)
    planner_node = build_planner_node(registry)
    task_progress_node = create_task_progressor()
    team_lead_node = build_team_lead_node(registry)
    investigation_agent_node = build_investigation_agent_node(registry)
    study_agent_node = build_study_agent_node(registry)
    assay_agent_node = build_assay_agent_node(registry)
    process_agent_node = build_process_agent_node(registry)
    protocol_agent_node = build_protocol_agent_node(registry)
    factor_agent_node = build_factor_agent_node(registry)
    source_agent_node = build_source_agent_node(registry)
    sample_agent_node = build_sample_agent_node(registry)
    material_agent_node = build_material_agent_node(registry)
    datafile_agent_node = build_datafile_agent_node(registry)

    # Build and compile graph
    app = build_graph(
        isa_overview_node,
        doc_overview_node,
        task_decomposer_node,
        task_progress_node,
        planner_node,
        search_document_caller,
        team_lead_node,
        investigation_agent_node,
        study_agent_node,
        assay_agent_node,
        process_agent_node,
        protocol_agent_node,
        factor_agent_node,
        source_agent_node,
        sample_agent_node,
        material_agent_node,
        datafile_agent_node,
    )

    return app

def run_stateful_graph_workflow(document_path: str) -> ISAExtractionState | None:
    app = create_stateful_graph_workflow(document_path)

    doc_manager = index_document(document_path)
    document = doc_manager.document

    # Initialize empty state
    initial_state = ISAExtractionState(
        document = document,
        messages=[],
        draft_isa_metadata={},
        evidence={},
        validation_result={},
        isa_mental_model="",
        document_overview="",
        tasks=[],
        last_action=None
    )

    # Provide a thread_id so checkpoints are recorded under this session
    run_id = uuid.uuid4()
    thread_id = f"isa-extract-{run_id}"
    config = RunnableConfig(recursion_limit=9000000, configurable={"thread_id": thread_id})

    stream_messages(run_id, app, initial_state, config=config)

    # After streaming completes, fetch the final state
    try:
        state_snapshot = app.get_state(config)
        write_final_state(state_snapshot, run_id)
        return state_snapshot
    except Exception as e:
        print(f"Failed to retrieve final state: {e}")
        return None