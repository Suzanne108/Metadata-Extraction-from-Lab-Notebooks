import json
from pathlib import Path
from typing import Any

from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src.graphs.nodes.base_node import LLMGraphNode
from src.graphs.types.Node import GraphComponentId
from src.graphs.types.Action import WorkflowAction
from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt


class AssemblyAdvisorNode(LLMGraphNode):
    """LLM node that proposes how to stitch DTOs into a proper ISA graph.

    The advisor receives:
    - The original document text
    - JSON dumps of all created DTO registries

    It returns a text recommendation and writes it to `output/<doc_stem>_assembly_advice.txt`.
    """

    def __init__(self, model: ChatOpenAI, system_prompt: str, inference_prompt_template) -> None:
        super().__init__(
            node_id=GraphComponentId.ASSEMBLY_ADVISOR.value,
            model=model,
            system_prompt=system_prompt,
            inference_prompt_template=inference_prompt_template,
        )

    def _document_stem(self, state: ISAExtractionState) -> str:
        try:
            doc = state.get("document")
            first = (doc[0] if isinstance(doc, list) and doc else doc)
            meta = getattr(first, "metadata", {}) or {}
            src = meta.get("source") or meta.get("path")
            if src:
                return Path(src).stem
        except (AttributeError, TypeError):
            pass
        return "isa_advice"

    def _dump_registry(self, reg: dict) -> str:
        """Convert registry of DTOs to a JSON string for prompt context."""
        def _to_dict(obj: Any) -> dict:
            try:
                return obj.model_dump()
            except (AttributeError, TypeError):
                try:
                    return obj.dict()
                except (AttributeError, TypeError):
                    return {}

        serializable = {k: _to_dict(v) for k, v in (reg or {}).items()}
        try:
            return json.dumps(serializable, ensure_ascii=False)
        except (TypeError, ValueError):
            return "{}"

    def _dump_single(self, dto: Any) -> str:
        if dto is None:
            return "{}"
        try:
            return json.dumps(dto.model_dump(), ensure_ascii=False)
        except (AttributeError, TypeError):
            try:
                return json.dumps(dto.dict(), ensure_ascii=False)
            except (AttributeError, TypeError):
                return "{}"

    def run(self, state: ISAExtractionState) -> dict:
        # Gather full document text
        raw_doc = state.get("document")
        full_text = ""
        try:
            if isinstance(raw_doc, list):
                full_text = "\n\n".join([getattr(p, "page_content", str(p)) for p in raw_doc])
            else:
                full_text = getattr(raw_doc, "page_content", str(raw_doc))
        except (AttributeError, TypeError):
            full_text = str(raw_doc)

        # Build JSON context for all registries
        inv_json = self._dump_single(state.get("dto_investigation"))
        studies_json = self._dump_registry(state.get("registry_studies") or {})
        assays_json = self._dump_registry(state.get("registry_assays") or {})
        data_json = self._dump_registry(state.get("registry_data") or {})
        processes_json = self._dump_registry(state.get("registry_processes") or {})
        protocols_json = self._dump_registry(state.get("registry_protocols") or {})
        factors_json = self._dump_registry(state.get("registry_factors") or {})
        factor_values_json = self._dump_registry(state.get("registry_factor_values") or {})
        sources_json = self._dump_registry(state.get("registry_sources") or {})
        samples_json = self._dump_registry(state.get("registry_samples") or {})
        materials_json = self._dump_registry(state.get("registry_materials") or {})

        prompt = render_prompt(
            self.inference_prompt_template,
            {
                "document_text": full_text,
                "investigation": inv_json,
                "studies": studies_json,
                "assays": assays_json,
                "datafiles": data_json,
                "processes": processes_json,
                "protocols": protocols_json,
                "factors": factors_json,
                "factor_values": factor_values_json,
                "sources": sources_json,
                "samples": samples_json,
                "materials": materials_json,
            },
        )

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        response = self.model.invoke(messages)

        advice_text = response.content if hasattr(response, "content") else str(response)

        # Persist advice to output text file
        stem = self._document_stem(state)
        out_dir = Path.cwd() / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_assembly_advice.txt"
        try:
            out_path.write_text(advice_text, encoding="utf-8")
            msg = f"NODE: ASSEMBLY_ADVISOR; status=success; wrote={out_path}"
        except OSError as e:
            msg = f"NODE: ASSEMBLY_ADVISOR; status=error; write_failed={e}"

        return {
            "assembly_advice": advice_text,
            "messages": [AIMessage(content=msg)],
            "last_action": WorkflowAction.ASSEMBLY_ADVICE,
        }

