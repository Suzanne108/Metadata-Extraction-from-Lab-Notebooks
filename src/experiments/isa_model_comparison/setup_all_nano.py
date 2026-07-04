"""Model-comparison setup: all model-driven nodes use gpt-5-nano."""

from src.graphs.types.Node import GraphComponentId

from src.experiments.experiment_setup import ExperimentSetup


NODE_MODEL_NAMES: dict[str, str] = {
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


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="model-all-nano",
    description="Model comparison variant: all model-driven nodes run with gpt-5-nano.",
    node_model_names=NODE_MODEL_NAMES,
)
