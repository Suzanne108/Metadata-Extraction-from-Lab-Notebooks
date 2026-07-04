from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentSetup:
    """Declarative setup for one ISA extraction experiment variant."""

    variant_name: str
    description: str
    node_model_names: dict[str, str]
    node_prompt_configs: dict[str, dict[str, str]] = field(default_factory=dict)
