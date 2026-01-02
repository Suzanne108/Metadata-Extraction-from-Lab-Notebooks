from src.state.v2 import ISAExtractionStateV2
from src.tools import get_schema, validate_ISA_schema
from langchain.messages import AIMessage, ToolMessage


def read_schema(state: ISAExtractionStateV2):

    schema_name = state['orchestrator_decision'].schema
    
    # Fetch schema
    if schema_name is not None:
        schema = get_schema(schema_name)

        schema_cache = dict(state.get("schema_cache", {}))

        message = f"Succesfully fetched {schema_name.value}"

        # Add to the schema cache (return a new dict, not the return value of update())
        schema_cache[schema_name.value] = schema
        return {
            "schema_cache": schema_cache,
            "messages": state.get("messages", []) + [ToolMessage(content=message, tool_call_id="read_schema")]
        }
     
    else:
        message = "Define which schema to fetch"
        
        return {
            "messages": state.get("messages", []) + [ToolMessage(content= message, tool_call_id="read_schema")]
        }
    

def create_search_document_caller(search_document_tool: callable):

    def search_document_caller(state: ISAExtractionStateV2):
        query = state['orchestrator_decision'].query

        # Call the search document tool
        results = search_document_tool(query)

        message = f"Retrieved {len(results)} documents for query: {query}"

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



    