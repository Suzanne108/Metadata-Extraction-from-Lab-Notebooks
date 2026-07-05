"""Model-comparison setup: all model-driven nodes use gpt-5-nano."""

from src.model import get_chat_model

from src.experiments.experiment_setup import ExperimentSetup


DEFAULT_MODEL = get_chat_model(model_name="gpt-5-nano")


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="model-all-nano",
    description="Model comparison variant: all model-driven nodes run with gpt-5-nano.",
    default_model=DEFAULT_MODEL,
)
