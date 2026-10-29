"""Model-comparison setup: all model-driven nodes use gpt-5-mini."""

from src.model import get_chat_model

from src.experiments.experiment_setup import ExperimentSetup


DEFAULT_MODEL = get_chat_model(model_name="gpt-5-mini")


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="model-all-mini",
    description="Model comparison variant: all model-driven nodes run with gpt-5-mini.",
    default_model=DEFAULT_MODEL,
)
