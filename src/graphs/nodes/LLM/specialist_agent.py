import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from src.graphs.nodes.base_node import LLMGraphNode
from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt
from src.graphs.utils import ensure_dto_uid_and_register

#TODO: create a more proper factory pattern for creating specialist agents. This is a temporary solution to avoid code duplication. 


class SpecialistAgentNode(LLMGraphNode):
    """LLM node that produces one specialist DTO from instruction/evidence.

    Parameters
    - model: chat model configured with structured output for the target DTO type
    - system_prompt: the specialist's system instruction
    - inference_prompt_template: the specialist's inference prompt (expects instruction, latest_evidence)
    - dto_state_key: state key to store the resulting DTO (e.g., "dto_investigation", "dto_study")
    - agent_label: human-friendly label for logs/messages (e.g., "INVESTIGATION_AGENT", "ASSAY_AGENT")
    """

    # Mapping from dto state key to context variables and the registry keys to load
    SPECIALIST_CONTEXT_BY_DTO_KEY: dict[str, dict[str, str]] = {
        # Higher-level should know potential lower-level references
        "dto_investigation": {
        },
        "dto_study": {
        },
        "dto_assay": {
            "existing_assays": "registry_assays",
            "existing_datafiles": "registry_data",
            "existing_samples": "registry_samples",
            "existing_materials": "registry_materials",
            "existing_processes": "registry_processes",
        },
        "dto_process": {
            "existing_processes": "registry_processes",
            "existing_protocols": "registry_protocols",
            "existing_materials": "registry_materials",
            "existing_samples": "registry_samples",
            "existing_sources": "registry_sources",
        },
        "dto_protocol": {
            "existing_protocols": "registry_protocols",
        },
        "dto_factor": {
            "existing_factors": "registry_factors",
        },
        "dto_source": {
        },
        "dto_sample": {
        },
        "dto_material": {
        },
        "dto_data": {
            "existing_datafiles": "registry_data",
            "existing_processes": "registry_processes",
        },
    }

    def __init__(
        self,
        model: ChatOpenAI,
        system_prompt: str,
        inference_prompt_template,
        dto_state_key: str,
        agent_label: str,
    ) -> None:
        super().__init__(
            node_id=agent_label,
            model=model,
            system_prompt=system_prompt,
            inference_prompt_template=inference_prompt_template,
        )
        self.dto_state_key = dto_state_key
        self.agent_label = agent_label

    def _dump_registry(self, reg: dict) -> str:
        """Helper to convert registry of ISA objects to JSON string for prompt context. Handles non-serializable objects."""
        def _to_dict(obj: Any) -> dict:
            try:
                return obj.model_dump()
            except Exception:
                try:
                    return obj.dict()
                except Exception:
                    return {}
        serializable = {k: _to_dict(v) for k, v in reg.items()}
        try:
            return json.dumps(serializable, ensure_ascii=False)
        except Exception:
            return "{}"

    def _format_latest_evidence(self, latest_evidence: Any) -> str:
        """Convert latest_evidence list into a readable string for prompts.

        Handles lists of LangChain Documents, dicts with page_content, or plain strings.
        """
        if not latest_evidence:
            return "None."

        # Normalize to list
        if not isinstance(latest_evidence, list):
            latest_evidence = [latest_evidence]

        formatted_chunks = []
        for idx, item in enumerate(latest_evidence, start=1):
            # LangChain Document or similar object with page_content attribute
            text = getattr(item, "page_content", None)
            # Dict-style document
            if text is None and isinstance(item, dict):
                text = item.get("page_content") or item.get("content")
            # Fallback: string representation
            if text is None:
                text = str(item)

            text = str(text).strip()
            if not text:
                continue
            formatted_chunks.append(f"{idx}. {text}")

        if not formatted_chunks:
            return "None."

        return "\n".join(formatted_chunks)

    def run(self, state: ISAExtractionState) -> dict:
        decision = state.get("team_lead_decision")
        instruction = getattr(decision, "instruction", None)
        raw_latest_evidence = state.get("latest_evidence") or []

        # Build context variables of relevant existing objects
        context_map = self.SPECIALIST_CONTEXT_BY_DTO_KEY.get(self.dto_state_key, {})
        context_vars: dict[str, str] = {}
        for var_name, reg_key in context_map.items():
            reg = state.get(reg_key) or {}
            context_vars[var_name] = self._dump_registry(reg)

        latest_evidence = self._format_latest_evidence(raw_latest_evidence)
        print(f"passed evidence to agent: {latest_evidence}")

        prompt = render_prompt(
            self.inference_prompt_template,
            {
                "instruction": instruction,
                "latest_evidence": latest_evidence,
                **context_vars,
            },
        )

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        response: Any = self.model.invoke(messages)

        # Upsert primary DTO and any nested DTOs into per-type registries (overwrite policy)
        updates = ensure_dto_uid_and_register(state, response)
        updates[self.dto_state_key] = response

        updates["messages"] = [
            AIMessage(content=f"NODE: UPDATE_DRAFT; agent: {self.agent_label}; status: success"),
            AIMessage(content=f"Created or updated {self.dto_state_key}"),
        ]
        # Signal to planner guardrails that a draft update just occurred
        updates["last_action"] = "UPDATE_DRAFT"

        return updates

