from pathlib import Path

from src.state import ISAExtractionState
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.graphs.nodes.base_node import LLMGraphNode
from src.graphs.types.GraphComponentId import GraphComponentId
from src.prompts.renderer import render_prompt


class DocumentOverviewNode(LLMGraphNode):
    """LLM node that generates a document overview from source text and ISA context."""

    def __init__(
        self,
        model: ChatOpenAI,
        system_prompt: str,
        inference_prompt_template: str,
        isa_mental_model_path: Path | None = None,
    ) -> None:
        super().__init__(
            node_id=GraphComponentId.DOC_OVERVIEW.value,
            model=model,
            system_prompt=system_prompt,
            inference_prompt_template=inference_prompt_template,
        )
        self.isa_mental_model_path = isa_mental_model_path

    def run(self, state: ISAExtractionState) -> dict:
        # Prefer a fixed ISA mental model loaded from disk, falling back to
        # whatever is present in the state for backwards compatibility.
        mental_model = ""
        if self.isa_mental_model_path is not None:
            try:
                if self.isa_mental_model_path.exists():
                    mental_model = self.isa_mental_model_path.read_text(encoding="utf-8")
            except Exception:
                # If reading fails for any reason, fall back to state
                mental_model = state.get("isa_mental_model", "") or ""
        else:
            mental_model = state.get("isa_mental_model", "") or ""

        raw_doc = state.get("document")

        # document can be List[Document] or a single Document; handle both
        full_text = ""
        try:
            if isinstance(raw_doc, list):
                full_text = "\n\n".join([getattr(p, "page_content", str(p)) for p in raw_doc])
            else:
                full_text = getattr(raw_doc, "page_content", str(raw_doc))
        except Exception:
            full_text = str(raw_doc)

        prompt = render_prompt(
            self.inference_prompt_template,
            {
                "mental_model": mental_model,
                "text": full_text
            },
        )

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        response = self.model.invoke(messages)

        overview_text = response.content if hasattr(response, "content") else str(response)
        return {
            "document_overview": overview_text,
            "messages": [AIMessage(content=overview_text)],
        }
