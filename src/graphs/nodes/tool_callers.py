from src.state.v3 import ISAExtractionStateV3
from src.tools import get_schema, validate_ISA_schema
from langchain.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
    

def create_search_document_caller(search_document_tool: StructuredTool):

    def search_document_caller(state: ISAExtractionStateV3):
        if not state['orchestrator_decision'].query:
            return{
                "messages": state.get("messages", []) + [ToolMessage(content="No query provided for document search.", tool_call_id="search_document")]
            }
        query = state['orchestrator_decision'].query

        # Call the search document tool
        # in src/graphs/nodes/tool_callers.py -> create_search_document_caller
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
            "messages": state.get("messages", []) + [ToolMessage(content=message, tool_call_id="search_document")]
        }
    
    return search_document_caller

def validate(state: ISAExtractionStateV3):
    draft_isa_metadata = state.get("draft_isa_metadata")
    if not draft_isa_metadata:
        message = "No draft ISA metadata to validate."
        return {
            "messages": state.get("messages", []) + [ToolMessage(content=message, tool_call_id="validation")]
        }

    # Call validation tool directly
    if hasattr(validate_ISA_schema, "invoke"):
        validation_result = validate_ISA_schema.invoke({"isa": draft_isa_metadata})
    elif hasattr(validate_ISA_schema, "run"):
        # Fallback: some tools expose .run that accepts positional/keyword args
        validation_result = validate_ISA_schema.run(draft_isa_metadata)
    elif callable(validate_ISA_schema):
        # Last resort if it's a plain function
        validation_result = validate_ISA_schema(draft_isa_metadata)
    else:
        raise TypeError("validate_ISA_schema is not invokable; expected a LangChain tool or callable")

    # Add validation result to messages
    messages = list(state.get("messages", []))
    import json as _json
    messages.append(ToolMessage(
        content=f"NODE: VALIDATE; result: {_json.dumps(validation_result, ensure_ascii=False)}",
        tool_call_id="validation"
    ))

    return {"messages": messages, "validation_result": validation_result}



    