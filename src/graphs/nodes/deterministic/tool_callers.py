from src.state import ISAExtractionState
from src.tools import validate_ISA_schema
from langchain.messages import ToolMessage
from langchain_core.tools import StructuredTool
import json as _json
from src.graphs.nodes.base_node import DeterministicGraphNode
from src.graphs.types.Action import WorkflowAction
from src.graphs.types.Node import GraphComponentId
    

class SearchDocumentCallerNode(DeterministicGraphNode):
    """Deterministic node that executes semantic search against the document index."""

    def __init__(self, search_document_tool: StructuredTool) -> None:
        super().__init__(node_id=GraphComponentId.SEARCH_DOCUMENT.value)
        self.search_document_tool = search_document_tool

    def run(self, state: ISAExtractionState):
        if not state['planner_decision'].query:
            return{
                "messages": [ToolMessage(content="No query provided for document search.", tool_call_id="search_document")]
            }
        query = state['planner_decision'].query

        # Call the search document tool
        if hasattr(self.search_document_tool, "run"):
            results = self.search_document_tool.run(query, k=state['planner_decision'].k or 1)
        elif callable(self.search_document_tool):
            results = self.search_document_tool(query, k=state['planner_decision'].k or 1)
        else:
            raise TypeError("search_document_tool is not callable and has no .run() method")
        try:
            retrieved_count = len(results) if hasattr(results, "__len__") else 1
        except Exception:
            retrieved_count = 1
        message = f"NODE: SEARCH_DOCUMENT; query: {query}; k: {state['planner_decision'].k or 1}; retrieved_count: {retrieved_count}"

        # Copy the evidence dict and update it; dict.update() returns None so avoid using it directly
        evidence = dict(state.get("evidence", {}))
        evidence[query] = results

        return {
            "evidence": evidence,
            "latest_evidence": results,
            "latest_evidence_query": query,
            "messages": [ToolMessage(content=message, tool_call_id="search_document")],
            "last_action": WorkflowAction.SEARCH_DOCUMENT
        }

#TODO: not sure the name of this file is appropriate
#TODO: not sure the validate function should be here

def validate(state: ISAExtractionState):
    draft_isa_metadata = state.get("draft_isa_metadata")
    if not draft_isa_metadata:
        message = "No draft ISA metadata to validate."
        return {
            "messages": state.get("messages", []) + [ToolMessage(content=message, tool_call_id="validation")],
            "last_action": WorkflowAction.VALIDATE
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

    # Track consecutive failures and last failing path
    failures = int(state.get("consecutive_validation_failures") or 0)
    last_path = None
    try:
        is_valid = bool((validation_result or {}).get("valid") is True)
    except Exception:
        is_valid = False

    if not is_valid:
        failures += 1
        last_path = (validation_result or {}).get("json_path")
    else:
        failures = 0
        last_path = None

    # Add validation result to messages and state
    return {
        "messages": [ToolMessage(
            content=f"NODE: VALIDATE; result: {_json.dumps(validation_result, ensure_ascii=False)}",
            tool_call_id="validation"
            )],
            "validation_result": validation_result,
            "consecutive_validation_failures": failures,
            "last_validation_json_path": last_path,
            "last_action": WorkflowAction.VALIDATE
        }



    