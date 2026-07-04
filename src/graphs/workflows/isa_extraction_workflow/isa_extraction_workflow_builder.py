from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph

from src.graphs.core import ConditionalRoute, WorkflowContext, WorkflowGraphBuilder
from src.graphs.nodes.LLM.document_overview import DocumentOverviewNode
from src.graphs.nodes.LLM.task_decomposer import TaskDecomposerNode
from src.graphs.nodes.LLM.planner import PlannerNode
from src.graphs.nodes.routing_function import routing_function, team_lead_routing
from src.graphs.nodes.LLM.assembly_advisor import AssemblyAdvisorNode
from src.graphs.nodes.deterministic.investigation_export import InvestigationExportNode
from src.state import ISAExtractionState
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.graphs.nodes.deterministic.task_progress import TaskProgressNode
from src.graphs.types.NextStep import NextStepLLM
from src.graphs.types.Node import GraphComponentId
from src.task import TaskList
from src.graphs.nodes.LLM.isa_team_lead import ISATeamLeadNode
from src.graphs.nodes.LLM.specialist_agent import SpecialistAgentNode
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

SPECIALIST_NODE_CONFIGS: list[tuple[GraphComponentId, Any, str, str, str]] = [
    (GraphComponentId.INVESTIGATION_CREATOR, InvestigationDTO, "investigation_specialist", "dto_investigation", "INVESTIGATION_AGENT"),
    (GraphComponentId.STUDY_CREATOR, StudyDTO, "study_specialist", "dto_study", "STUDY_AGENT"),
    (GraphComponentId.ASSAY_CREATOR, AssayDTO, "assay_specialist", "dto_assay", "ASSAY_AGENT"),
    (GraphComponentId.PROCESS_CREATOR, ProcessDTO, "process_specialist", "dto_process", "PROCESS_AGENT"),
    (GraphComponentId.PROTOCOL_CREATOR, ProtocolDTO, "protocol_specialist", "dto_protocol", "PROTOCOL_AGENT"),
    (GraphComponentId.FACTOR_CREATOR, FactorDTO, "factor_specialist", "dto_factor", "FACTOR_AGENT"),
    (GraphComponentId.SOURCE_CREATOR, SourceDTO, "source_specialist", "dto_source", "SOURCE_AGENT"),
    (GraphComponentId.SAMPLE_CREATOR, SampleDTO, "sample_specialist", "dto_sample", "SAMPLE_AGENT"),
    (GraphComponentId.MATERIAL_CREATOR, MaterialDTO, "material_specialist", "dto_material", "MATERIAL_AGENT"),
    (GraphComponentId.DATAFILE_CREATOR, DataFileDTO, "datafile_specialist", "dto_data", "DATAFILE_AGENT"),
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
    return PlannerNode(model=model_planner, system_prompt=planner_system_prompt, policy_prompt=planner_policy_prompt)

def build_doc_overview_node(registry: PromptRegistry, model: Any):
    """Create ISA overview and document overview nodes with their prompts."""
    model_llm = model

    # Document Overview
    doc_overview_system = render_prompt(registry.get_prompt("document_overview", "v1-system"), {})
    doc_overview_infer = registry.get_prompt("document_overview", "v1-inference")
    isa_mental_model_path = Path.cwd() / "instructions" / "isa_mental_model.txt"
    doc_overview_node = DocumentOverviewNode(
        model=model_llm,
        system_prompt=doc_overview_system,
        inference_prompt_template=doc_overview_infer,
        isa_mental_model_path=isa_mental_model_path,
    )

    return doc_overview_node


def build_task_decomposer_node(registry: PromptRegistry, model: Any):
    """Create the task decomposer node using a dedicated structured-output model."""
    task_decomp_system = render_prompt(registry.get_prompt("task_decomposer", "v1-system"), {})
    task_decomp_infer = registry.get_prompt("task_decomposer", "v1-inference")
    model_task_decomposer = model.with_structured_output(TaskList)
    return TaskDecomposerNode(model=model_task_decomposer, system_prompt=task_decomp_system, inference_prompt_template=task_decomp_infer)


def build_team_lead_node(registry: PromptRegistry, model: Any):
    """Create ISA creation team lead node with prompts."""
    model = model.with_structured_output(TeamLeadDecision)
    system_prompt = render_prompt(registry.get_prompt("isa_team_lead", "v1-system"), {})
    infer_prompt = registry.get_prompt("isa_team_lead", "v1-inference")

    return ISATeamLeadNode(model=model, system_prompt=system_prompt, inference_prompt_template=infer_prompt)


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
    return SpecialistAgentNode(
        model=model,
        system_prompt=system_prompt,
        inference_prompt_template=infer_prompt,
        dto_state_key=dto_state_key,
        agent_label=agent_label,
    )


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
    return InvestigationExportNode()

def build_assembly_advisor_node(registry: PromptRegistry, model: Any):
    """Create the assembly advisor node with its prompts."""
    system_prompt = render_prompt(registry.get_prompt("assembly_advisor", "v1-system"), {})
    infer_prompt = registry.get_prompt("assembly_advisor", "v1-inference")
    return AssemblyAdvisorNode(model=model, system_prompt=system_prompt, inference_prompt_template=infer_prompt)

def build_graph(nodes: dict[str, Any], context: WorkflowContext) -> StateGraph:
    """Add nodes, wire edges, and compile the graph using shared context."""
    linear_edges = [
        ("START", GraphComponentId.DOC_OVERVIEW.value),
        (GraphComponentId.DOC_OVERVIEW.value, GraphComponentId.TASK_DECOMPOSER.value),
        (GraphComponentId.TASK_DECOMPOSER.value, GraphComponentId.TASK_PROGRESS.value),
        (GraphComponentId.TASK_PROGRESS.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.INVESTIGATION_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.STUDY_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.ASSAY_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.PROCESS_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.PROTOCOL_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.FACTOR_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.SOURCE_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.SAMPLE_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.MATERIAL_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.DATAFILE_CREATOR.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.SEARCH_DOCUMENT.value, GraphComponentId.PLANNER_AGENT.value),
        (GraphComponentId.ASSEMBLY_ADVISOR.value, GraphComponentId.EXPORT.value),
        (GraphComponentId.EXPORT.value, "END"),
    ]

    conditional_routes = [
        ConditionalRoute(
            source=GraphComponentId.PLANNER_AGENT.value,
            route_fn=routing_function,
            route_map={
                GraphComponentId.TEAM_LEAD.value: GraphComponentId.TEAM_LEAD.value,
                GraphComponentId.SEARCH_DOCUMENT.value: GraphComponentId.SEARCH_DOCUMENT.value,
                GraphComponentId.TASK_PROGRESS.value: GraphComponentId.TASK_PROGRESS.value,
                GraphComponentId.ASSEMBLY_ADVISOR.value: GraphComponentId.ASSEMBLY_ADVISOR.value,
            },
        ),
        ConditionalRoute(
            source=GraphComponentId.TEAM_LEAD.value,
            route_fn=team_lead_routing,
            route_map={
                GraphComponentId.INVESTIGATION_CREATOR.value: GraphComponentId.INVESTIGATION_CREATOR.value,
                GraphComponentId.STUDY_CREATOR.value: GraphComponentId.STUDY_CREATOR.value,
                GraphComponentId.ASSAY_CREATOR.value: GraphComponentId.ASSAY_CREATOR.value,
                GraphComponentId.PROCESS_CREATOR.value: GraphComponentId.PROCESS_CREATOR.value,
                GraphComponentId.PROTOCOL_CREATOR.value: GraphComponentId.PROTOCOL_CREATOR.value,
                GraphComponentId.FACTOR_CREATOR.value: GraphComponentId.FACTOR_CREATOR.value,
                GraphComponentId.SOURCE_CREATOR.value: GraphComponentId.SOURCE_CREATOR.value,
                GraphComponentId.SAMPLE_CREATOR.value: GraphComponentId.SAMPLE_CREATOR.value,
                GraphComponentId.MATERIAL_CREATOR.value: GraphComponentId.MATERIAL_CREATOR.value,
                GraphComponentId.DATAFILE_CREATOR.value: GraphComponentId.DATAFILE_CREATOR.value,
                GraphComponentId.PLANNER_AGENT.value: GraphComponentId.PLANNER_AGENT.value,
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
        GraphComponentId.DOC_OVERVIEW.value: build_doc_overview_node(
            registry,
            _resolve_node_model(context, resolved_node_models, GraphComponentId.DOC_OVERVIEW.value, default_model),
        ),
        GraphComponentId.TASK_DECOMPOSER.value: build_task_decomposer_node(
            registry,
            _resolve_node_model(context, resolved_node_models, GraphComponentId.TASK_DECOMPOSER.value, default_model),
        ),
        GraphComponentId.TASK_PROGRESS.value: TaskProgressNode(),
        GraphComponentId.PLANNER_AGENT.value: build_planner_node(
            registry,
            _resolve_node_model(context, resolved_node_models, GraphComponentId.PLANNER_AGENT.value, default_model),
        ),
        GraphComponentId.SEARCH_DOCUMENT.value: context.search_document_caller,
        GraphComponentId.TEAM_LEAD.value: build_team_lead_node(
            registry,
            _resolve_node_model(context, resolved_node_models, GraphComponentId.TEAM_LEAD.value, default_model),
        ),
        GraphComponentId.ASSEMBLY_ADVISOR.value: build_assembly_advisor_node(
            registry,
            _resolve_node_model(context, resolved_node_models, GraphComponentId.ASSEMBLY_ADVISOR.value, default_model),
        ),
        GraphComponentId.EXPORT.value: build_export_node(),
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

