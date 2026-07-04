"""Model-comparison setup: all model-driven nodes use gpt-5-mini."""

from src.graphs.types.Node import GraphComponentId

from src.experiments.experiment_setup import ExperimentSetup


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


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="model-all-mini",
    description="Model comparison variant: all model-driven nodes run with gpt-5-mini.",
    node_model_names=NODE_MODEL_NAMES,
)
