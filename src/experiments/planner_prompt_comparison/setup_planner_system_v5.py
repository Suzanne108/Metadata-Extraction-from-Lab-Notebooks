"""Planner-prompt setup: fixed models, planner system prompt version v5-system."""

from src.graphs.types.Node import GraphComponentId

from src.experiments.experiment_setup import ExperimentSetup

#TODO: make sure there is a default model configuration defined to avoid duplicated code

# Keep models fixed to isolate planner system-prompt effect.
NODE_MODEL_NAMES: dict[str, str] = {
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

NODE_PROMPT_CONFIGS: dict[str, dict[str, str]] = {
    GraphComponentId.PLANNER_AGENT.value: {
        "prompt_key": "planner",
        "system_prompt_version": "v5-system",
        "policy_prompt_version": "v3-policy",
    },
}


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="planner-system-v5",
    description="Planner prompt comparison: fixed models, planner system=v5-system, policy=v3-policy.",
    node_model_names=NODE_MODEL_NAMES,
    node_prompt_configs=NODE_PROMPT_CONFIGS,
)
