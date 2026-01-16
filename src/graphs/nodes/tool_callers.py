from src.state import ISAExtractionState
from src.tools import validate_ISA_schema
from langchain.messages import ToolMessage
from langchain_core.tools import StructuredTool
import json as _json
    

def create_search_document_caller(search_document_tool: StructuredTool):

    def search_document_caller(state: ISAExtractionState):
        if not state['orchestrator_decision'].query:
            return{
                "messages": [ToolMessage(content="No query provided for document search.", tool_call_id="search_document")]
            }
        query = state['orchestrator_decision'].query

        # Call the search document tool
        if hasattr(search_document_tool, "run"):
            results = search_document_tool.run(query, k=state['orchestrator_decision'].k or 1)
        elif callable(search_document_tool):
            results = search_document_tool(query, k=state['orchestrator_decision'].k or 1)
        else:
            raise TypeError("search_document_tool is not callable and has no .run() method")
        try:
            retrieved_count = len(results) if hasattr(results, "__len__") else 1
        except Exception:
            retrieved_count = 1
        message = f"NODE: SEARCH_DOCUMENT; query: {query}; k: {state['orchestrator_decision'].k or 1}; retrieved_count: {retrieved_count}"

        # Copy the evidence dict and update it; dict.update() returns None so avoid using it directly
        evidence = dict(state.get("evidence", {}))
        evidence[query] = results

        return {
            "evidence": evidence,
            "latest_evidence": results,
            "messages": [ToolMessage(content=message, tool_call_id="search_document")],
            "last_action": "SEARCH_DOCUMENT"
        }
    
    return search_document_caller

def validate(state: ISAExtractionState):
    draft_isa_metadata = state.get("draft_isa_metadata")
    if not draft_isa_metadata:
        message = "No draft ISA metadata to validate."
        return {
            "messages": state.get("messages", []) + [ToolMessage(content=message, tool_call_id="validation")],
            "last_action": "VALIDATE"
        }
    
    # TODO: Fix validation calling code

    validation_result = None
    # Call validation tool directly
    if hasattr(validate_ISA_schema, "invoke"):
        validation_result = validate_ISA_schema.invoke({"isa": draft_isa_metadata})
    elif hasattr(validate_ISA_schema, "run"):
        # Fallback: some tools expose .run that accepts positional/keyword args
        validation_result = validate_ISA_schema.run(draft_isa_metadata)
    else:
        validation_result = "Was not able to call validation tool."

    # Add validation result to messages
    return {
        "messages": [ToolMessage(
            content=f"NODE: VALIDATE; result: {_json.dumps(validation_result, ensure_ascii=False)}",
            tool_call_id="validation"
            )],
            "validation_result": validation_result,
            "last_action": "VALIDATE"
        }



    