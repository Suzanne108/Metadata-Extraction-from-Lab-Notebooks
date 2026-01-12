from src.state.v2 import ISAExtractionStateV2
from src.tools import get_schema, validate_ISA_schema
from langchain.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
    

def create_search_document_caller(search_document_tool: StructuredTool):

    def search_document_caller(state: ISAExtractionStateV2):
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
        message = f"For query: {query} \n\n retrieved the following content: \n {results}"

        # Copy the evidence dict and update it; dict.update() returns None so avoid using it directly
        evidence = dict(state.get("evidence", {}))
        evidence[query] = results

        return {
            "evidence": evidence,
            "latest_evidence": results,
            "messages": state.get("messages", []) + [ToolMessage(content=message, tool_call_id="search_document")]
        }
    
    return search_document_caller

def validate(state: ISAExtractionStateV2):
    draft_isa_metadata = state.get("draft_isa_metadata")
    if not draft_isa_metadata:
        message = "No draft ISA metadata to validate."
        return {
            "messages": state.get("messages", []) + [ToolMessage(content=message, tool_call_id="validation")]
        }

    # Call validation tool directly
    validation_result = validate_ISA_schema(draft_isa_metadata)

    # Add validation result to messages
    messages = list(state.get("messages", []))
    messages.append(ToolMessage(
        content=str(validation_result),
        tool_call_id="validation"
    ))

    return {"messages": messages, "validation_result": validation_result}



    