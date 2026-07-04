from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.graphs.types.Node import GraphComponentId
from src.graphs.workflows.isa_extraction_workflow.final_state import handle_final_state
from src.graphs.workflows.isa_extraction_workflow.initial_state import create_initial_state
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import (
    create_stateful_graph_workflow,
)
from src.graphs.workflows.workflow_runner import run_graph_workflow


MODEL_DRIVEN_NODES: list[GraphComponentId] = [
    GraphComponentId.DOC_OVERVIEW,
    GraphComponentId.TASK_DECOMPOSER,
    GraphComponentId.PLANNER_AGENT,
    GraphComponentId.TEAM_LEAD,
    GraphComponentId.INVESTIGATION_CREATOR,
    GraphComponentId.STUDY_CREATOR,
    GraphComponentId.ASSAY_CREATOR,
    GraphComponentId.PROCESS_CREATOR,
    GraphComponentId.PROTOCOL_CREATOR,
    GraphComponentId.FACTOR_CREATOR,
    GraphComponentId.SOURCE_CREATOR,
    GraphComponentId.SAMPLE_CREATOR,
    GraphComponentId.MATERIAL_CREATOR,
    GraphComponentId.DATAFILE_CREATOR,
    GraphComponentId.ASSEMBLY_ADVISOR,
]


ALL_MINI_NODE_MODEL_NAMES: dict[str, str] = {
    GraphComponentId.DOC_OVERVIEW.value: "gpt-5-mini",
    GraphComponentId.TASK_DECOMPOSER.value: "gpt-5-mini",
    GraphComponentId.PLANNER_AGENT.value: "gpt-5-mini",
    GraphComponentId.TEAM_LEAD.value: "gpt-5-mini",
    GraphComponentId.INVESTIGATION_CREATOR.value: "gpt-5-mini",
    GraphComponentId.STUDY_CREATOR.value: "gpt-5-mini",
    GraphComponentId.ASSAY_CREATOR.value: "gpt-5-mini",
    GraphComponentId.PROCESS_CREATOR.value: "gpt-5-mini",
    GraphComponentId.PROTOCOL_CREATOR.value: "gpt-5-mini",
    GraphComponentId.FACTOR_CREATOR.value: "gpt-5-mini",
    GraphComponentId.SOURCE_CREATOR.value: "gpt-5-mini",
    GraphComponentId.SAMPLE_CREATOR.value: "gpt-5-mini",
    GraphComponentId.MATERIAL_CREATOR.value: "gpt-5-mini",
    GraphComponentId.DATAFILE_CREATOR.value: "gpt-5-mini",
    GraphComponentId.ASSEMBLY_ADVISOR.value: "gpt-5-mini",
}

ALL_NANO_NODE_MODEL_NAMES: dict[str, str] = {
    GraphComponentId.DOC_OVERVIEW.value: "gpt-5-nano",
    GraphComponentId.TASK_DECOMPOSER.value: "gpt-5-nano",
    GraphComponentId.PLANNER_AGENT.value: "gpt-5-nano",
    GraphComponentId.TEAM_LEAD.value: "gpt-5-nano",
    GraphComponentId.INVESTIGATION_CREATOR.value: "gpt-5-nano",
    GraphComponentId.STUDY_CREATOR.value: "gpt-5-nano",
    GraphComponentId.ASSAY_CREATOR.value: "gpt-5-nano",
    GraphComponentId.PROCESS_CREATOR.value: "gpt-5-nano",
    GraphComponentId.PROTOCOL_CREATOR.value: "gpt-5-nano",
    GraphComponentId.FACTOR_CREATOR.value: "gpt-5-nano",
    GraphComponentId.SOURCE_CREATOR.value: "gpt-5-nano",
    GraphComponentId.SAMPLE_CREATOR.value: "gpt-5-nano",
    GraphComponentId.MATERIAL_CREATOR.value: "gpt-5-nano",
    GraphComponentId.DATAFILE_CREATOR.value: "gpt-5-nano",
    GraphComponentId.ASSEMBLY_ADVISOR.value: "gpt-5-nano",
}

HYBRID_NODE_MODEL_NAMES: dict[str, str] = {
    GraphComponentId.DOC_OVERVIEW.value: "gpt-5-nano",
    GraphComponentId.TASK_DECOMPOSER.value: "gpt-5-mini",
    GraphComponentId.PLANNER_AGENT.value: "gpt-5-mini",
    GraphComponentId.TEAM_LEAD.value: "gpt-5-nano",
    GraphComponentId.INVESTIGATION_CREATOR.value: "gpt-5-nano",
    GraphComponentId.STUDY_CREATOR.value: "gpt-5-nano",
    GraphComponentId.ASSAY_CREATOR.value: "gpt-5-nano",
    GraphComponentId.PROCESS_CREATOR.value: "gpt-5-nano",
    GraphComponentId.PROTOCOL_CREATOR.value: "gpt-5-nano",
    GraphComponentId.FACTOR_CREATOR.value: "gpt-5-nano",
    GraphComponentId.SOURCE_CREATOR.value: "gpt-5-nano",
    GraphComponentId.SAMPLE_CREATOR.value: "gpt-5-nano",
    GraphComponentId.MATERIAL_CREATOR.value: "gpt-5-nano",
    GraphComponentId.DATAFILE_CREATOR.value: "gpt-5-nano",
    GraphComponentId.ASSEMBLY_ADVISOR.value: "gpt-5-nano",
}


def create_chat_model(model_name: str) -> ChatOpenAI:
    """Build a chat model instance for experiments."""
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)


def build_node_models(node_model_names: dict[str, str]) -> dict[str, ChatOpenAI]:
    """Build per-node model objects from an explicit node->model-name map."""
    model_cache: dict[str, ChatOpenAI] = {}
    node_models: dict[str, ChatOpenAI] = {}
    for node_name in [node.value for node in MODEL_DRIVEN_NODES]:
        model_name = node_model_names[node_name]
        if model_name not in model_cache:
            model_cache[model_name] = create_chat_model(model_name)
        node_models[node_name] = model_cache[model_name]
    return node_models


def run_experiment_variant(
    variant_name: str,
    node_model_names: dict[str, str],
    document_path: Path,
) -> str:
    """Run one extraction variant and return the run identifier and final state."""
    node_models = build_node_models(node_model_names)

    initial_state = create_initial_state(str(document_path))
    app = create_stateful_graph_workflow(
        document_path=str(document_path),
        node_models=node_models,
    )
    app, config, run_id = run_graph_workflow(
        app=app,
        initial_state=initial_state,
        run_prefix=f"isa-extract-{variant_name}",
        stream=False,
    )
    handle_final_state(app, config, run_id)

    return str(run_id)


def main() -> None:
    """Run three workflow variants: mini-all, nano-all, and hybrid."""
    document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf"

    mini_run_id = run_experiment_variant(
        variant_name="all-mini",
        node_model_names=ALL_MINI_NODE_MODEL_NAMES,
        document_path=document_path,
    )
    print(f"[all-mini] run_id={mini_run_id}")

    nano_run_id = run_experiment_variant(
        variant_name="all-nano",
        node_model_names=ALL_NANO_NODE_MODEL_NAMES,
        document_path=document_path,
    )
    print(f"[all-nano] run_id={nano_run_id}")

    hybrid_run_id = run_experiment_variant(
        variant_name="hybrid-mini-planner-taskdecomposer",
        node_model_names=HYBRID_NODE_MODEL_NAMES,
        document_path=document_path,
    )
    print(f"[hybrid-mini-planner-taskdecomposer] run_id={hybrid_run_id}")


if __name__ == "__main__":
    main()
