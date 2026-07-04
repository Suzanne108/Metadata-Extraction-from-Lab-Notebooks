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
from src.model import get_chat_model
from src.prompts.renderer import render_prompt
from src.graphs.nodes.deterministic.task_progress import TaskProgressNode
from src.graphs.types.NextStep import NextStepLLM
from src.graphs.types.Node import GraphComponentId
from src.task import TaskList
from src.graphs.nodes.LLM.isa_team_lead import ISATeamLeadNode
from src.graphs.nodes.LLM.specialist_agent import SpecialistAgentNode
from src.graphs.types.TeamLeadDecision import TeamLeadDecision
from src.graphs.workflows.isa_extraction_workflow.node_prompt_config import (
    DEFAULT_NODE_PROMPT_CONFIGS,
    SPECIALIST_NODE_CONFIGS,
)

class ISAExtractionWorkflowBuilder:
    """Builder object for configuring and creating the ISA extraction graph workflow."""

    def __init__(
        self,
        document_path: str | None = None,
        *,
        node_models: dict[str, Any] | None = None,
        node_model_names: dict[str, str] | None = None,
        default_model: Any | None = None,
        default_model_name: str | None = None,
        node_prompt_configs: dict[str, dict[str, str]] | None = None,
        context: WorkflowContext | None = None,
    ) -> None:
        self.context = context or WorkflowContext.for_isa_extraction(document_path=document_path)
        self.registry = self.context.prompt_registry
        self.node_models = node_models or {}
        self.node_model_names = node_model_names or {}
        self.default_model = default_model
        self.default_model_name = default_model_name
        self.node_prompt_configs = node_prompt_configs or {}
        self._model_cache_by_name: dict[str, Any] = {}

    def with_node_models(self, node_models: dict[str, Any]) -> "ISAExtractionWorkflowBuilder":
        """Override explicit per-node model objects."""
        self.node_models = node_models
        return self

    def with_node_model_names(self, node_model_names: dict[str, str]) -> "ISAExtractionWorkflowBuilder":
        """Override per-node model names."""
        self.node_model_names = node_model_names
        return self

    def with_node_prompt_configs(self, node_prompt_configs: dict[str, dict[str, str]]) -> "ISAExtractionWorkflowBuilder":
        """Override per-node prompt settings."""
        self.node_prompt_configs = node_prompt_configs
        return self

    def _resolve_prompt_config(self, node_name: str) -> dict[str, str]:
        base_config = dict(DEFAULT_NODE_PROMPT_CONFIGS.get(node_name, {}))
        base_config.update(self.node_prompt_configs.get(node_name, {}))
        return base_config

    def _resolve_prompt_key(self, node_name: str) -> str:
        prompt_config = self._resolve_prompt_config(node_name)
        configured_prompt_key = prompt_config.get("prompt_key")
        if not configured_prompt_key:
            raise KeyError(f"No prompt_key configured for node '{node_name}'.")
        return configured_prompt_key

    def _render_system_prompt(self, node_name: str) -> str:
        prompt_config = self._resolve_prompt_config(node_name)
        prompt_name = self._resolve_prompt_key(node_name)
        version = prompt_config.get("system_prompt_version", "v1-system")
        return render_prompt(self.registry.get_prompt(prompt_name, version), {})

    def _get_inference_prompt(self, node_name: str):
        prompt_config = self._resolve_prompt_config(node_name)
        prompt_name = self._resolve_prompt_key(node_name)
        version = prompt_config.get("inference_prompt_version", "v1-inference")
        return self.registry.get_prompt(prompt_name, version)

    def _render_policy_prompt(self, node_name: str) -> str:
        prompt_config = self._resolve_prompt_config(node_name)
        prompt_name = self._resolve_prompt_key(node_name)
        version = prompt_config.get("policy_prompt_version", "v3-policy")
        return render_prompt(self.registry.get_prompt(prompt_name, version), {})

    def _model_from_name(self, model_name: str) -> Any:
        if model_name not in self._model_cache_by_name:
            self._model_cache_by_name[model_name] = get_chat_model(model_name=model_name)
        return self._model_cache_by_name[model_name]

    def _resolve_node_model(self, node_name: str) -> Any:
        if node_name in self.node_models:
            return self.node_models[node_name]
        if node_name in self.node_model_names:
            return self._model_from_name(self.node_model_names[node_name])
        if self.default_model is not None:
            return self.default_model
        if self.default_model_name is not None:
            return self._model_from_name(self.default_model_name)
        return self.context.provider_registry.get_chat_model()

    def _build_planner_node(self) -> PlannerNode:
        node_name = GraphComponentId.PLANNER_AGENT.value
        model_planner = self._resolve_node_model(node_name).with_structured_output(NextStepLLM)
        planner_system_prompt = self._render_system_prompt(node_name)
        planner_policy_prompt = self._render_policy_prompt(node_name)
        return PlannerNode(model=model_planner, system_prompt=planner_system_prompt, policy_prompt=planner_policy_prompt)

    def _build_doc_overview_node(self) -> DocumentOverviewNode:
        node_name = GraphComponentId.DOC_OVERVIEW.value
        isa_mental_model_path = Path.cwd() / "instructions" / "isa_mental_model.txt"
        return DocumentOverviewNode(
            model=self._resolve_node_model(node_name),
            system_prompt=self._render_system_prompt(node_name),
            inference_prompt_template=self._get_inference_prompt(node_name),
            isa_mental_model_path=isa_mental_model_path,
        )

    def _build_task_decomposer_node(self) -> TaskDecomposerNode:
        node_name = GraphComponentId.TASK_DECOMPOSER.value
        model_task_decomposer = self._resolve_node_model(node_name).with_structured_output(TaskList)
        return TaskDecomposerNode(
            model=model_task_decomposer,
            system_prompt=self._render_system_prompt(node_name),
            inference_prompt_template=self._get_inference_prompt(node_name),
        )

    def _build_team_lead_node(self) -> ISATeamLeadNode:
        node_name = GraphComponentId.TEAM_LEAD.value
        model = self._resolve_node_model(node_name).with_structured_output(TeamLeadDecision)
        return ISATeamLeadNode(
            model=model,
            system_prompt=self._render_system_prompt(node_name),
            inference_prompt_template=self._get_inference_prompt(node_name),
        )

    def _build_specialist_node(
        self,
        node_name: str,
        dto_cls: Any,
        dto_state_key: str,
        agent_label: str,
    ) -> SpecialistAgentNode:
        model = self._resolve_node_model(node_name).with_structured_output(dto_cls)
        return SpecialistAgentNode(
            model=model,
            system_prompt=self._render_system_prompt(node_name),
            inference_prompt_template=self._get_inference_prompt(node_name),
            dto_state_key=dto_state_key,
            agent_label=agent_label,
        )

    def _build_specialist_nodes(self) -> dict[str, Any]:
        specialist_nodes: dict[str, Any] = {}
        for specialist_node_id, dto_cls, _, dto_state_key, agent_label in SPECIALIST_NODE_CONFIGS:
            specialist_nodes[specialist_node_id.value] = self._build_specialist_node(
                node_name=specialist_node_id.value,
                dto_cls=dto_cls,
                dto_state_key=dto_state_key,
                agent_label=agent_label,
            )
        return specialist_nodes

    @staticmethod
    def _build_export_node() -> InvestigationExportNode:
        return InvestigationExportNode()

    def _build_assembly_advisor_node(self) -> AssemblyAdvisorNode:
        node_name = GraphComponentId.ASSEMBLY_ADVISOR.value
        return AssemblyAdvisorNode(
            model=self._resolve_node_model(node_name),
            system_prompt=self._render_system_prompt(node_name),
            inference_prompt_template=self._get_inference_prompt(node_name),
        )

    def _build_nodes(self) -> dict[str, Any]:
        nodes: dict[str, Any] = {
            GraphComponentId.DOC_OVERVIEW.value: self._build_doc_overview_node(),
            GraphComponentId.TASK_DECOMPOSER.value: self._build_task_decomposer_node(),
            GraphComponentId.TASK_PROGRESS.value: TaskProgressNode(),
            GraphComponentId.PLANNER_AGENT.value: self._build_planner_node(),
            GraphComponentId.SEARCH_DOCUMENT.value: self.context.search_document_caller,
            GraphComponentId.TEAM_LEAD.value: self._build_team_lead_node(),
            GraphComponentId.ASSEMBLY_ADVISOR.value: self._build_assembly_advisor_node(),
            GraphComponentId.EXPORT.value: self._build_export_node(),
        }
        nodes.update(self._build_specialist_nodes())
        return nodes

    @staticmethod
    def _linear_edges() -> list[tuple[str, str]]:
        return [
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

    @staticmethod
    def _conditional_routes() -> list[ConditionalRoute]:
        return [
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

    def _compile_graph(self, nodes: dict[str, Any]) -> StateGraph:
        graph_builder = WorkflowGraphBuilder(ISAExtractionState, checkpointer=self.context.checkpointer)
        return (
            graph_builder
            .add_nodes(nodes)
            .add_edges(self._linear_edges())
            .add_conditional_routes(self._conditional_routes())
            .compile()
        )

    def build(self) -> StateGraph:
        """Build and compile the configured ISA extraction workflow."""
        nodes = self._build_nodes()
        return self._compile_graph(nodes)

def create_stateful_graph_workflow(
    document_path: str = None,
    node_models: dict[str, Any] | None = None,
    node_model_names: dict[str, str] | None = None,
    node_prompt_configs: dict[str, dict[str, str]] | None = None,
    default_model: Any | None = None,
    default_model_name: str | None = None,
) -> StateGraph:
    """Create and compile ISA extraction workflow with optional per-node overrides.

    Args:
        document_path: Path to the source document.
        node_models: Mapping of node name to model object.
        node_model_names: Mapping of node name to model name.
        node_prompt_configs: Mapping of node name to prompt configuration.
        default_model: Optional fallback model object.
        default_model_name: Optional fallback model name.
    """
    return ISAExtractionWorkflowBuilder(
        document_path=document_path,
        node_models=node_models,
        node_model_names=node_model_names,
        default_model=default_model,
        default_model_name=default_model_name,
        node_prompt_configs=node_prompt_configs,
    ).build()

