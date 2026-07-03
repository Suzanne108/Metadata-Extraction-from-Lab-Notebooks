from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph

from src.graphs.core import ConditionalRoute, WorkflowContext, WorkflowGraphBuilder
from src.graphs.nodes.document_overview import create_document_overview
from src.graphs.nodes.task_decomposer import create_task_decomposer
from src.graphs.nodes.planner import create_planner
from src.graphs.nodes.routing_function import routing_function, team_lead_routing
from src.graphs.nodes.assembly_advisor import create_assembly_advisor
from src.graphs.nodes.investigation_export import create_investigation_export_node
from src.state import ISAExtractionState
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.nodes.task_progress import create_task_progressor
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

#TODO: does this config need so many parameters? Can we simplify it? For example, the dto_state_key is already in the dto_cls, and the agent_label is just a human-readable version of the node type. Maybe we can just use a single config that maps Node to DTO class, and derive

SPECIALIST_NODE_CONFIGS: list[tuple[Node, Any, str, str, str]] = [
    (Node.INVESTIGATION_CREATOR, InvestigationDTO, "investigation_specialist", "dto_investigation", "INVESTIGATION_AGENT"),
    (Node.STUDY_CREATOR, StudyDTO, "study_specialist", "dto_study", "STUDY_AGENT"),
    (Node.ASSAY_CREATOR, AssayDTO, "assay_specialist", "dto_assay", "ASSAY_AGENT"),
    (Node.PROCESS_CREATOR, ProcessDTO, "process_specialist", "dto_process", "PROCESS_AGENT"),
    (Node.PROTOCOL_CREATOR, ProtocolDTO, "protocol_specialist", "dto_protocol", "PROTOCOL_AGENT"),
    (Node.FACTOR_CREATOR, FactorDTO, "factor_specialist", "dto_factor", "FACTOR_AGENT"),
    (Node.SOURCE_CREATOR, SourceDTO, "source_specialist", "dto_source", "SOURCE_AGENT"),
    (Node.SAMPLE_CREATOR, SampleDTO, "sample_specialist", "dto_sample", "SAMPLE_AGENT"),
    (Node.MATERIAL_CREATOR, MaterialDTO, "material_specialist", "dto_material", "MATERIAL_AGENT"),
    (Node.DATAFILE_CREATOR, DataFileDTO, "datafile_specialist", "dto_data", "DATAFILE_AGENT"),
]

def _resolve_node_model(
    context: WorkflowContext,
    node_models: dict[str, Any],
    node_name: str,
    default_model: Any | None,
) -> Any:
    """Resolve model for a specific node with client override precedence."""
    if node_name in node_models:
        return node_models[node_name]
    if default_model is not None:
        return default_model
    return context.provider_registry.get_chat_model()


def build_planner_node(registry: PromptRegistry, model: Any):
    """Create the planner node with structured output and prompts."""
    model_planner = model.with_structured_output(NextStepLLM)
    planner_system_prompt = render_prompt(registry.get_prompt("planner", "v5-system"), {})
    planner_policy_prompt = render_prompt(registry.get_prompt("planner", "v3-policy"), {})
    return create_planner(model_planner, planner_system_prompt, planner_policy_prompt)

def build_doc_overview_node(registry: PromptRegistry, model: Any):
    """Create ISA overview and document overview nodes with their prompts."""
    model_llm = model

    # Document Overview
    doc_overview_system = render_prompt(registry.get_prompt("document_overview", "v1-system"), {})
    doc_overview_infer = registry.get_prompt("document_overview", "v1-inference")
    isa_mental_model_path = Path.cwd() / "instructions" / "isa_mental_model.txt"
    doc_overview_node = create_document_overview(
        model_llm,
        doc_overview_system,
        doc_overview_infer,
        isa_mental_model_path,
    )

    return doc_overview_node


def build_task_decomposer_node(registry: PromptRegistry, model: Any):
    """Create the task decomposer node using a dedicated structured-output model."""
    task_decomp_system = render_prompt(registry.get_prompt("task_decomposer", "v1-system"), {})
    task_decomp_infer = registry.get_prompt("task_decomposer", "v1-inference")
    model_task_decomposer = model.with_structured_output(TaskList)
    return create_task_decomposer(model_task_decomposer, task_decomp_system, task_decomp_infer)


def build_team_lead_node(registry: PromptRegistry, model: Any):
    """Create ISA creation team lead node with prompts."""
    model = model.with_structured_output(TeamLeadDecision)
    system_prompt = render_prompt(registry.get_prompt("isa_team_lead", "v1-system"), {})
    infer_prompt = registry.get_prompt("isa_team_lead", "v1-inference")

    return create_isa_team_lead(model, system_prompt, infer_prompt)


def build_specialist_node(
    registry: PromptRegistry,
    model: Any,
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
    model = model.with_structured_output(dto_cls)
    system_prompt = render_prompt(registry.get_prompt(prompt_key, "v1-system"), {})
    infer_prompt = registry.get_prompt(prompt_key, "v1-inference")
    return create_specialist_agent(model, system_prompt, infer_prompt, dto_state_key, agent_label)


def build_specialist_nodes(
    registry: PromptRegistry,
    context: WorkflowContext,
    node_models: dict[str, Any],
    default_model: Any | None,
) -> dict[str, Any]:
    """Build all specialist nodes from a single declarative config list."""
    specialist_nodes: dict[str, Any] = {}
    for node_type, dto_cls, prompt_key, dto_state_key, agent_label in SPECIALIST_NODE_CONFIGS:
        model_for_node = _resolve_node_model(
            context=context,
            node_models=node_models,
            node_name=node_type.value,
            default_model=default_model,
        )
        specialist_nodes[node_type.value] = build_specialist_node(
            registry=registry,
            model=model_for_node,
            dto_cls=dto_cls,
            prompt_key=prompt_key,
            dto_state_key=dto_state_key,
            agent_label=agent_label,
        )
    return specialist_nodes

def build_export_node():
    """Build the simple export node that writes ISA-JSON."""
    return create_investigation_export_node()

def build_assembly_advisor_node(registry: PromptRegistry, model: Any):
    """Create the assembly advisor node with its prompts."""
    system_prompt = render_prompt(registry.get_prompt("assembly_advisor", "v1-system"), {})
    infer_prompt = registry.get_prompt("assembly_advisor", "v1-inference")
    return create_assembly_advisor(model, system_prompt, infer_prompt)

def build_graph(nodes: dict[str, Any], context: WorkflowContext) -> StateGraph:
    """Add nodes, wire edges, and compile the graph using shared context."""
    linear_edges = [
        ("START", Node.DOC_OVERVIEW.value),
        (Node.DOC_OVERVIEW.value, Node.TASK_DECOMPOSER.value),
        (Node.TASK_DECOMPOSER.value, Node.TASK_PROGRESS.value),
        (Node.TASK_PROGRESS.value, Node.PLANNER_AGENT.value),
        (Node.INVESTIGATION_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.STUDY_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.ASSAY_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.PROCESS_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.PROTOCOL_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.FACTOR_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.SOURCE_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.SAMPLE_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.MATERIAL_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.DATAFILE_CREATOR.value, Node.PLANNER_AGENT.value),
        (Node.SEARCH_DOCUMENT.value, Node.PLANNER_AGENT.value),
        (Node.ASSEMBLY_ADVISOR.value, Node.EXPORT.value),
        (Node.EXPORT.value, "END"),
    ]

    conditional_routes = [
        ConditionalRoute(
            source=Node.PLANNER_AGENT.value,
            route_fn=routing_function,
            route_map={
                Node.TEAM_LEAD.value: Node.TEAM_LEAD.value,
                Node.SEARCH_DOCUMENT.value: Node.SEARCH_DOCUMENT.value,
                Node.TASK_PROGRESS.value: Node.TASK_PROGRESS.value,
                Node.ASSEMBLY_ADVISOR.value: Node.ASSEMBLY_ADVISOR.value,
            },
        ),
        ConditionalRoute(
            source=Node.TEAM_LEAD.value,
            route_fn=team_lead_routing,
            route_map={
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
        ),
    ]

    builder = WorkflowGraphBuilder(ISAExtractionState, checkpointer=context.checkpointer)
    return (
        builder
        .add_nodes(nodes)
        .add_edges(linear_edges)
        .add_conditional_routes(conditional_routes)
        .compile()
    )

def create_stateful_graph_workflow(
    document_path: str = None,
    node_models: dict[str, Any] | None = None,
    default_model: Any | None = None,
) -> StateGraph:
    """Create and compile ISA extraction workflow with optional per-node model overrides.

    Args:
        document_path: Path to the source document.
        node_models: Mapping of node name to model instance. Keys should match
            values from ``Node.<X>.value``.
        default_model: Optional fallback model for nodes not present in
            ``node_models``.
    """
    context = WorkflowContext.for_isa_extraction(document_path=document_path)
    registry = context.prompt_registry
    resolved_node_models = node_models or {}

    nodes: dict[str, Any] = {
        Node.DOC_OVERVIEW.value: build_doc_overview_node(
            registry,
            _resolve_node_model(context, resolved_node_models, Node.DOC_OVERVIEW.value, default_model),
        ),
        Node.TASK_DECOMPOSER.value: build_task_decomposer_node(
            registry,
            _resolve_node_model(context, resolved_node_models, Node.TASK_DECOMPOSER.value, default_model),
        ),
        Node.TASK_PROGRESS.value: create_task_progressor(),
        Node.PLANNER_AGENT.value: build_planner_node(
            registry,
            _resolve_node_model(context, resolved_node_models, Node.PLANNER_AGENT.value, default_model),
        ),
        Node.SEARCH_DOCUMENT.value: context.search_document_caller,
        Node.TEAM_LEAD.value: build_team_lead_node(
            registry,
            _resolve_node_model(context, resolved_node_models, Node.TEAM_LEAD.value, default_model),
        ),
        Node.ASSEMBLY_ADVISOR.value: build_assembly_advisor_node(
            registry,
            _resolve_node_model(context, resolved_node_models, Node.ASSEMBLY_ADVISOR.value, default_model),
        ),
        Node.EXPORT.value: build_export_node(),
    }
    nodes.update(
        build_specialist_nodes(
            registry=registry,
            context=context,
            node_models=resolved_node_models,
            default_model=default_model,
        )
    )

    return build_graph(nodes=nodes, context=context)

