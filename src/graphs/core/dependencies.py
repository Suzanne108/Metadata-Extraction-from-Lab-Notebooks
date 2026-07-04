from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver

from src.graphs.nodes.deterministic.tool_callers import SearchDocumentCallerNode
from src.graphs.utils import index_document
from src.model import get_chat_model, get_embedding_model
from src.prompts.registry import PromptRegistry
from src.tools import create_doc_getter, create_schema_getter


class ProviderRegistry:
    """Caches provider instances and supports dynamic model injection by key."""

    def __init__(
        self,
        chat_model_factory: Callable[[], Any] | None = None,
        embedding_model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._chat_model_factory = chat_model_factory or get_chat_model
        self._embedding_model_factory = embedding_model_factory or get_embedding_model
        self._chat_models_by_key: dict[str, Any] = {}
        self._default_chat_model: Any | None = None
        self._embedding_model: Any | None = None

    def set_chat_model(self, key: str, model: Any) -> None:
        """Register an explicit chat model for a caller-defined key."""
        self._chat_models_by_key[key] = model

    def get_chat_model(self, key: str | None = None, structured_output: Any | None = None) -> Any:
        """Return chat model by key (if provided) or fallback to shared default model."""
        if key is not None and key in self._chat_models_by_key:
            base_model = self._chat_models_by_key[key]
        else:
            if self._default_chat_model is None:
                self._default_chat_model = self._chat_model_factory()
            base_model = self._default_chat_model

        if structured_output is not None:
            return base_model.with_structured_output(structured_output)
        return base_model

    def get_embedding_model(self) -> Any:
        """Return a cached embedding model."""
        if self._embedding_model is None:
            self._embedding_model = self._embedding_model_factory()
        return self._embedding_model


@dataclass(frozen=True)
class WorkflowContext:
    """Shared dependencies used while assembling and compiling workflows."""

    prompt_registry: PromptRegistry
    provider_registry: ProviderRegistry
    search_document_caller: Any
    schema_getter_tools: Any
    checkpointer: Any

    @classmethod
    def for_isa_extraction(
        cls,
        document_path: str | None = None,
        prompt_registry: PromptRegistry | None = None,
        provider_registry: ProviderRegistry | None = None,
        checkpointer: Any | None = None,
    ) -> "WorkflowContext":
        """Create the standard dependency context for ISA extraction workflows."""
        doc_manager = index_document(document_path=document_path)
        search_document_tool = create_doc_getter(doc_manager)
        search_document_caller = SearchDocumentCallerNode(search_document_tool)
        schema_getter_tools = create_schema_getter()

        return cls(
            prompt_registry=prompt_registry or PromptRegistry(),
            provider_registry=provider_registry or ProviderRegistry(),
            search_document_caller=search_document_caller,
            schema_getter_tools=schema_getter_tools,
            checkpointer=checkpointer or MemorySaver(),
        )
