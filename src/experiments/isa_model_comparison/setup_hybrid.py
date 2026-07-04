"""Model-comparison setup: hybrid allocation across gpt-5-mini and gpt-5-nano."""

from src.graphs.types.Node import GraphComponentId

from src.experiments.experiment_setup import ExperimentSetup


NODE_MODEL_NAMES: dict[str, str] = {
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


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="model-hybrid",
    description="Model comparison variant: task decomposition and planning use gpt-5-mini, others use gpt-5-nano.",
    node_model_names=NODE_MODEL_NAMES,
)
