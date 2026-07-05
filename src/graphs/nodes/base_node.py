from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseGraphNode(ABC):
    """Common callable contract for all graph node implementations."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    def __call__(self, state: Any) -> dict:
        return self.run(state)

    @abstractmethod
    def run(self, state: Any) -> dict:
        """Execute node logic and return a state update dictionary."""


class LLMGraphNode(BaseGraphNode):
    """Base class for nodes that invoke an LLM."""

    def __init__(
        self,
        node_id: str,
        model: Any,
        system_prompt: str,
        inference_prompt_template: Any,
        policy_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(node_id)
        self.model = model
        self.system_prompt = system_prompt
        self.inference_prompt_template = inference_prompt_template
        self.policy_prompt = policy_prompt


class DeterministicGraphNode(BaseGraphNode):
    """Base class for nodes with deterministic/scripted state transitions."""
