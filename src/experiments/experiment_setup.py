from dataclasses import dataclass, field
from typing import Any

from src.graphs.types.GraphComponentId import GraphComponentId


@dataclass(frozen=True)
class ExperimentSetup:
    """Declarative setup for one ISA extraction experiment variant."""

    variant_name: str
    description: str
    node_models: dict[GraphComponentId, Any] = field(default_factory=dict)
    default_model: Any | None = None
    node_prompt_configs: dict[GraphComponentId, dict[str, str]] = field(default_factory=dict)

