

from src.state.v2 import ISAExtractionStateV2
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage

def create_ISA_updater(model: ChatOpenAI):

    #TODO Maybe this model should have the ISA schema's as a tool to call?

    def ISA_updater(state: ISAExtractionStateV2):

        draft_isa_metadata = state.get("draft_isa_metadata", {})
        focus = state['orchestrator_decision'].focus
        latest_evidence = state.get("latest_evidence", [])

        prompt = f"""
            You are an assistant that updates a draft ISA metadata object according to the official ISA JSON schemas.
            You have a tool available to retrieve schema definitions (callable as get_schema(schema_name) or equivalent) — always use it to validate and normalize updates.

            Inputs
            - draft_isa_metadata (JSON): {draft_isa_metadata}
            - focus (string): {focus}            # JSON path or schema field/section to update (e.g., "study.contacts" or "assays[0].measurement_type")
            - latest_evidence (list): {latest_evidence}

            Instructions
            1) Retrieve any ISA schema(s) required to validate the focused section using the get_schema tool.
            2) Apply information from `latest_evidence` to update ONLY the specified `focus` field or section inside `draft_isa_metadata`.
            - Do not change unrelated fields or top-level structure unless the schema requires a structural normalization.
            3) Validate and normalize all updated values to match schema types and constraints:
            - Use ontology term objects (identifier, term, source) when schema expects ontology annotations.
            - Convert simple strings to arrays or objects if the schema requires that shape.
            - Enforce required fields and data types for the updated section.
            4) Never introduce fields not defined by the ISA schemas. If evidence suggests new information that cannot map cleanly, put it into a schema-compliant field or include a clear warning in `messages`.
            5) If the evidence is ambiguous or insufficient to make a safe schema-compliant change, do not modify the draft — instead return the unchanged `draft_isa_metadata` and include a clarifying question or a validation warning in `messages`.
            6) Produce a concise change summary describing what was updated and any validation warnings.

            Output format (MUST be valid JSON only)
            Return only the ISA metadata JSON object with the applied updates.

            Do not output any text outside the JSON. Begin processing now.
            """
        
        response = model.invoke(prompt)

        return {
            "draft_isa_metadata": response,
            "messages": state.get("messages", []) +  [AIMessage(content="Updated draft ISA metadata.")]
        }


    
    return ISA_updater


