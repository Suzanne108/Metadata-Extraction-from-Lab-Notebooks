"""Model-comparison setup: hybrid allocation across gpt-5-mini and gpt-5-nano."""

from src.graphs.types.GraphComponentId import GraphComponentId
from src.model import get_chat_model
from src.graphs.workflows.isa_extraction_workflow.node_prompt_config import DEFAULT_NODE_PROMPT_CONFIGS

from src.experiments.experiment_setup import ExperimentSetup


DEFAULT_MODEL = get_chat_model(model_name="gpt-5-nano")
NODE_MODELS = {
    GraphComponentId.DOC_OVERVIEW: get_chat_model(model_name="gpt-5-mini", reasoning_effort="high"),
    GraphComponentId.TASK_DECOMPOSER: get_chat_model(model_name="gpt-5-mini", reasoning_effort="medium"),
    GraphComponentId.ASSEMBLY_ADVISOR: get_chat_model(model_name="gpt-5-mini", reasoning_effort="medium")
}


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="model-hybrid",
    description="Model comparison variant: agents needing more reasoning use gpt-5-mini, others use gpt-5-nano.",
    node_models=NODE_MODELS,
    default_model=DEFAULT_MODEL,
    node_prompt_configs=DEFAULT_NODE_PROMPT_CONFIGS
)
