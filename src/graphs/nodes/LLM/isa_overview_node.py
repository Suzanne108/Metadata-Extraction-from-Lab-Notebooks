from typing import List
from langchain_core.documents import Document
from src.state import ISAExtractionState
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.graphs.nodes.base_node import LLMGraphNode
from src.graphs.types.GraphComponentId import GraphComponentId
from src.prompts.renderer import render_prompt
from src.graphs.utils import invoke_model_with_tools


class ISAOverviewNode(LLMGraphNode):
    """LLM node that produces an ISA mental model from the ISA abstract model document."""

    def __init__(
        self,
        model: ChatOpenAI,
        schema_getter_tools: list,
        system_prompt: str,
        inference_prompt_template: str,
        isa_overview_doc: List[Document],
    ) -> None:
        super().__init__(
            node_id=GraphComponentId.ISA_OVERVIEW.value,
            model=model,
            system_prompt=system_prompt,
            inference_prompt_template=inference_prompt_template,
        )
        self.schema_getter_tools = schema_getter_tools
        self.isa_overview_doc = isa_overview_doc

    def run(self, state: ISAExtractionState) -> dict:

        full_text = ""
        try:
            if isinstance(self.isa_overview_doc, list):
                full_text = "\n\n".join([getattr(p, "page_content", str(p)) for p in self.isa_overview_doc])
            else:
                full_text = getattr(self.isa_overview_doc, "page_content", str(self.isa_overview_doc))
        except Exception:
            full_text = str(self.isa_overview_doc)

        prompt = render_prompt(self.inference_prompt_template, {"ISA_abstract_model": full_text})
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]

        response = invoke_model_with_tools(self.model, messages, self.schema_getter_tools, max_iterations=20)

        mental_model = response.content if hasattr(response, "content") else str(response)
        return {
            "isa_mental_model": mental_model,
            "messages": [AIMessage(content=mental_model)]
        }

