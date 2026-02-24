import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt
from src.graphs.utils import ensure_dto_uid_and_register


def create_specialist_agent(
    model: ChatOpenAI,
    system_prompt: str,
    inference_prompt_template,
    dto_state_key: str,
    agent_label: str,
) -> callable:
    """Creates a generic specialist agent that produces a single DTO from instruction/evidence.

    Parameters
    - model: chat model configured with structured output for the target DTO type
    - system_prompt: the specialist's system instruction
    - inference_prompt_template: the specialist's inference prompt (expects instruction, latest_evidence)
    - dto_state_key: state key to store the resulting DTO (e.g., "dto_investigation", "dto_study")
    - agent_label: human-friendly label for logs/messages (e.g., "INVESTIGATION_AGENT", "ASSAY_AGENT")
    """

    def _dump_registry(reg: dict) -> str:
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

    # Mapping from dto state key to context variables and the registry keys to load
    SPECIALIST_CONTEXT_BY_DTO_KEY: dict[str, dict[str, str]] = {
        # Higher-level should know potential lower-level references
        "dto_investigation": {
            "existing_investigations": "registry_investigations",
            "existing_studies": "registry_studies",
        },
        "dto_study": {
            "existing_studies": "registry_studies",
            "existing_assays": "registry_assays",
            "existing_factors": "registry_factors",
            "existing_sources": "registry_sources",
            "existing_samples": "registry_samples",
            "existing_materials": "registry_materials",
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
            "existing_sources": "registry_sources",
            "existing_materials": "registry_materials",
        },
        "dto_sample": {
            "existing_samples": "registry_samples",
            "existing_sources": "registry_sources",
            "existing_materials": "registry_materials",
        },
        "dto_material": {
            "existing_materials": "registry_materials",
        },
        "dto_data": {
            "existing_datafiles": "registry_data",
            "existing_processes": "registry_processes",
        },
    }

    def specialist_agent(state: ISAExtractionState) -> dict:
        decision = state.get("team_lead_decision")
        instruction = getattr(decision, "instruction", None)
        latest_evidence = state.get("latest_evidence") or []

        # Build context variables of relevant existing objects
        context_map = SPECIALIST_CONTEXT_BY_DTO_KEY.get(dto_state_key, {})
        context_vars: dict[str, str] = {}
        for var_name, reg_key in context_map.items():
            reg = state.get(reg_key) or {}
            context_vars[var_name] = _dump_registry(reg)

        prompt = render_prompt(
            inference_prompt_template,
            {
                "instruction": instruction,
                "latest_evidence": latest_evidence,
                **context_vars,
            },
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response: Any = model.invoke(messages)

        print(response)

        # Upsert primary DTO and any nested DTOs into per-type registries (overwrite policy)
        updates = ensure_dto_uid_and_register(state, response)
        updates[dto_state_key] = response

        updates["messages"] = [
            AIMessage(content=f"NODE: UPDATE_DRAFT; agent: {agent_label}; status: success"),
            AIMessage(content=f"Created or updated {dto_state_key}"),
        ]
        # Signal to planner guardrails that a draft update just occurred
        updates["last_action"] = "UPDATE_DRAFT"

        return updates

    return specialist_agent
