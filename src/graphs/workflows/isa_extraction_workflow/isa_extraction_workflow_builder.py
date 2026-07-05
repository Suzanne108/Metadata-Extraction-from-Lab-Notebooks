from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from src.graphs.core import ConditionalRoute, WorkflowGraphBuilder
from src.graphs.nodes.LLM.document_overview_node import DocumentOverviewNode
from src.graphs.nodes.LLM.task_decomposer_node import TaskDecomposerNode
from src.graphs.nodes.LLM.planner_node import PlannerNode
from src.graphs.nodes.routing_function import planner_routing_function, team_lead_routing
from src.graphs.nodes.LLM.assembly_advisor_node import AssemblyAdvisorNode
from src.graphs.nodes.deterministic.investigation_export_node import InvestigationExportNode
from src.graphs.nodes.deterministic.tool_callers import SearchDocumentCallerNode
from src.state import ISAExtractionState
from src.model import get_chat_model
from src.prompts.renderer import render_prompt
from src.prompts.registry import PromptRegistry
from src.graphs.nodes.deterministic.task_progress_node import TaskProgressNode
from src.graphs.types.NextStep import NextStepLLM
from src.graphs.types.GraphComponentId import GraphComponentId
from src.task import TaskList
from src.tools import create_doc_getter
from src.graphs.nodes.LLM.isa_team_lead_node import ISATeamLeadNode
from src.graphs.nodes.LLM.specialist_agent_node import SpecialistAgentNode
from src.graphs.types.TeamLeadDecision import TeamLeadDecision
from src.graphs.workflows.isa_extraction_workflow.node_prompt_config import (
    DEFAULT_NODE_PROMPT_CONFIGS,
    SPECIALIST_NODE_CONFIGS,
)
from src.document_processing import DocumentManager

class ISAExtractionWorkflowBuilder:
    """Builder object for configuring and creating the ISA extraction graph workflow."""

    def __init__(
        self,
        document_manager: DocumentManager,
        node_models: dict[GraphComponentId, Any] | None = None,
        default_model: Any | None = None,
        node_prompt_configs: dict[GraphComponentId, dict[str, str]] | None = None,
    ) -> None:
        if document_manager.document is None:
            raise ValueError("document_manager must load a document before building the workflow.")

        self.search_document_tool = create_doc_getter(document_manager)
        self.registry = PromptRegistry()
        self.checkpointer = MemorySaver()
        self.node_models = node_models or {}
        self.default_model = default_model
        self.node_prompt_configs = node_prompt_configs or {}

    def with_node_models(self, node_models: dict[GraphComponentId, Any]) -> "ISAExtractionWorkflowBuilder":
        """Override explicit per-node model objects."""
        self.node_models = node_models
        return self

    def with_default_model(self, default_model: Any) -> "ISAExtractionWorkflowBuilder":
        """Override fallback model object."""
        self.default_model = default_model
        return self

    def with_node_prompt_configs(self, node_prompt_configs: dict[GraphComponentId, dict[str, str]]) -> "ISAExtractionWorkflowBuilder":
        """Override per-node prompt settings."""
        self.node_prompt_configs = node_prompt_configs
        return self

    def _resolve_prompt_config(self, node_id: GraphComponentId) -> dict[str, str]:
        base_config = dict(DEFAULT_NODE_PROMPT_CONFIGS.get(node_id, {}))
        base_config.update(self.node_prompt_configs.get(node_id, {}))
        return base_config

    def _resolve_prompt_key(self, node_id: GraphComponentId) -> str:
        prompt_config = self._resolve_prompt_config(node_id)
        configured_prompt_key = prompt_config.get("prompt_key")
        if not configured_prompt_key:
            raise KeyError(f"No prompt_key configured for node '{node_id.value}'.")
        return configured_prompt_key

    def _require_prompt_version(self, node_id: GraphComponentId, version_key: str) -> str:
        prompt_config = self._resolve_prompt_config(node_id)
        configured_version = prompt_config.get(version_key)
        if not configured_version:
            raise KeyError(
                f"No '{version_key}' configured for node '{node_id.value}'."
            )
        return configured_version

    def _render_system_prompt(self, node_id: GraphComponentId) -> str:
        prompt_name = self._resolve_prompt_key(node_id)
        version = self._require_prompt_version(node_id, "system_prompt_version")
        return render_prompt(self.registry.get_prompt(prompt_name, version), {})

    def _get_inference_prompt(self, node_id: GraphComponentId):
        prompt_name = self._resolve_prompt_key(node_id)
        version = self._require_prompt_version(node_id, "inference_prompt_version")
        return self.registry.get_prompt(prompt_name, version)

    def _render_policy_prompt(self, node_id: GraphComponentId) -> str:
        prompt_name = self._resolve_prompt_key(node_id)
        version = self._require_prompt_version(node_id, "policy_prompt_version")
        return render_prompt(self.registry.get_prompt(prompt_name, version), {})

    def _resolve_node_model(self, node_id: GraphComponentId) -> Any:
        if node_id in self.node_models:
            return self.node_models[node_id]
        if self.default_model is not None:
            return self.default_model
        return get_chat_model()

    def _build_planner_node(self) -> PlannerNode:
        node_id = GraphComponentId.PLANNER_AGENT
        model_planner = self._resolve_node_model(node_id).with_structured_output(NextStepLLM)
        planner_system_prompt = self._render_system_prompt(node_id)
        planner_inference_prompt = self._get_inference_prompt(node_id)
        planner_policy_prompt = self._render_policy_prompt(node_id)
        return PlannerNode(
            model=model_planner,
            system_prompt=planner_system_prompt,
            inference_prompt_template=planner_inference_prompt,
            policy_prompt=planner_policy_prompt,
        )

    def _build_doc_overview_node(self) -> DocumentOverviewNode:
        node_id = GraphComponentId.DOC_OVERVIEW
        isa_mental_model_path = Path.cwd() / "instructions" / "isa_mental_model.txt"
        return DocumentOverviewNode(
            model=self._resolve_node_model(node_id),
            system_prompt=self._render_system_prompt(node_id),
            inference_prompt_template=self._get_inference_prompt(node_id),
            isa_mental_model_path=isa_mental_model_path,
        )

    def _build_task_decomposer_node(self) -> TaskDecomposerNode:
        node_id = GraphComponentId.TASK_DECOMPOSER
        model_task_decomposer = self._resolve_node_model(node_id).with_structured_output(TaskList)
        return TaskDecomposerNode(
            model=model_task_decomposer,
            system_prompt=self._render_system_prompt(node_id),
            inference_prompt_template=self._get_inference_prompt(node_id),
        )

    def _build_team_lead_node(self) -> ISATeamLeadNode:
        node_id = GraphComponentId.TEAM_LEAD
        model = self._resolve_node_model(node_id).with_structured_output(TeamLeadDecision)
        return ISATeamLeadNode(
            model=model,
            system_prompt=self._render_system_prompt(node_id),
            inference_prompt_template=self._get_inference_prompt(node_id),
        )

    def _build_specialist_node(
        self,
        node_id: GraphComponentId,
        dto_cls: Any,
        base_name: str,
    ) -> SpecialistAgentNode:
        model = self._resolve_node_model(node_id).with_structured_output(dto_cls)
        return SpecialistAgentNode(
            model=model,
            system_prompt=self._render_system_prompt(node_id),
            inference_prompt_template=self._get_inference_prompt(node_id),
            base_name=base_name,
        )

    def _build_specialist_nodes(self) -> dict[str, Any]:
        specialist_nodes: dict[str, Any] = {}
        for specialist_config in SPECIALIST_NODE_CONFIGS:
            specialist_nodes[specialist_config.node_id.value] = self._build_specialist_node(
                node_id=specialist_config.node_id,
                dto_cls=specialist_config.dto_cls,
                base_name=specialist_config.base_name,
            )
        return specialist_nodes

    def _build_assembly_advisor_node(self) -> AssemblyAdvisorNode:
        node_id = GraphComponentId.ASSEMBLY_ADVISOR
        return AssemblyAdvisorNode(
            model=self._resolve_node_model(node_id),
            system_prompt=self._render_system_prompt(node_id),
            inference_prompt_template=self._get_inference_prompt(node_id),
        )

    def _build_nodes(self) -> dict[str, Any]:
        nodes: dict[str, Any] = {
            GraphComponentId.DOC_OVERVIEW.value: self._build_doc_overview_node(),
            GraphComponentId.TASK_DECOMPOSER.value: self._build_task_decomposer_node(),
            GraphComponentId.TASK_PROGRESS.value: TaskProgressNode(),
            GraphComponentId.PLANNER_AGENT.value: self._build_planner_node(),
            GraphComponentId.SEARCH_DOCUMENT.value: SearchDocumentCallerNode(self.search_document_tool),
            GraphComponentId.TEAM_LEAD.value: self._build_team_lead_node(),
            GraphComponentId.ASSEMBLY_ADVISOR.value: self._build_assembly_advisor_node(),
            GraphComponentId.EXPORT.value: InvestigationExportNode(),
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
                route_fn=planner_routing_function,
            ),
            ConditionalRoute(
                source=GraphComponentId.TEAM_LEAD.value,
                route_fn=team_lead_routing,
            ),
        ]

    def build(self) -> StateGraph:
        """Build and compile the configured ISA extraction workflow."""
        graph_builder = WorkflowGraphBuilder(ISAExtractionState, checkpointer=self.checkpointer)
        return (
            graph_builder
            .add_nodes(self._build_nodes())
            .add_edges(self._linear_edges())
            .add_conditional_routes(self._conditional_routes())
            .compile()
        )

