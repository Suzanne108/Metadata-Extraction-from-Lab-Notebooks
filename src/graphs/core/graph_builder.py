from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph


@dataclass(frozen=True)
class ConditionalRoute:
    """Declarative conditional routing specification for a node."""

    source: str
    route_fn: Callable[[dict], str]
    route_map: dict[str, str]


class WorkflowGraphBuilder:
    """Small helper around StateGraph to standardize build flow."""

    def __init__(self, state_type: type, checkpointer: Any | None = None) -> None:
        self._graph = StateGraph(state_type)
        self._checkpointer = checkpointer

    def add_nodes(self, nodes: dict[str, Callable]) -> "WorkflowGraphBuilder":
        for node_name, node_fn in nodes.items():
            self._graph.add_node(node_name, node_fn)
        return self

    def add_edges(self, edges: list[tuple[str, str]]) -> "WorkflowGraphBuilder":
        for source, target in edges:
            src = START if source == "START" else source
            dst = END if target == "END" else target
            self._graph.add_edge(src, dst)
        return self

    def add_conditional_routes(self, routes: list[ConditionalRoute]) -> "WorkflowGraphBuilder":
        for route in routes:
            self._graph.add_conditional_edges(route.source, route.route_fn, route.route_map)
        return self

    def compile(self) -> Any:
        if self._checkpointer is None:
            return self._graph.compile()
        return self._graph.compile(checkpointer=self._checkpointer)
