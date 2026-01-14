from src.state.v3 import ISAExtractionStateV3
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts.renderer import render_prompt


def create_document_overview(model: ChatOpenAI, system_prompt: str, inference_prompt_template: str):

    def document_overview(state: ISAExtractionStateV3) -> dict:
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
            inference_prompt_template,
            {
                "mental_model": mental_model,
                "text": full_text[:200000],  # safety cap
            },
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response = model.invoke(messages)

        overview_text = response.content if hasattr(response, "content") else str(response)
        return {
            "document_overview": overview_text,
            "messages": [AIMessage(content=overview_text)],
        }

    return document_overview
