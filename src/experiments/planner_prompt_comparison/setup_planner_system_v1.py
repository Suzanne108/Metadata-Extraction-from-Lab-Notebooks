"""Planner-prompt setup: fixed models, planner system prompt version v1-system."""

from src.graphs.types.GraphComponentId import GraphComponentId
from src.model import get_chat_model

from src.experiments.experiment_setup import ExperimentSetup

#TODO: make sure there is a default model configuration defined to avoid duplicated code

# Keep models fixed to isolate planner system-prompt effect.
DEFAULT_MODEL = get_chat_model(model_name="gpt-5-mini")

NODE_PROMPT_CONFIGS: dict[GraphComponentId, dict[str, str]] = {
    GraphComponentId.PLANNER_AGENT: {
        "prompt_key": "planner",
        "system_prompt_version": "v1-system",
        "policy_prompt_version": "v1-policy",
    },
}


EXPERIMENT_SETUP = ExperimentSetup(
    variant_name="planner-system-v1",
    description="Planner prompt comparison: fixed models, planner system=v1-system, policy=v1-policy.",
    default_model=DEFAULT_MODEL,
    node_prompt_configs=NODE_PROMPT_CONFIGS,
)
