"""Model-comparison setup: hybrid allocation across gpt-5-mini and gpt-5-nano."""

from src.graphs.types.GraphComponentId import GraphComponentId
from src.model import get_chat_model

from src.experiments.experiment_setup import ExperimentSetup


DEFAULT_MODEL = get_chat_model(model_name="gpt-5-nano")
NODE_MODELS = {
    GraphComponentId.TASK_DECOMPOSER: get_chat_model(model_name="gpt-5-mini"),
    GraphComponentId.PLANNER_AGENT: get_chat_model(model_name="gpt-5-mini"),
}


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="model-hybrid",
    description="Model comparison variant: task decomposition and planning use gpt-5-mini, others use gpt-5-nano.",
    node_models=NODE_MODELS,
    default_model=DEFAULT_MODEL,
)
