from langgraph.graph import StateGraph, START, END
from src.state.v2 import ISAExtractionStateV2
from src.document_processing import DocumentManager
from src.model import get_chat_model
from src.tools import create_doc_getter, create_schema_getter, validate_ISA_schema
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
import json
from pathlib import Path

def create_simple_workflow_v3(doc_manager: DocumentManager):
    """Creates a LangGraph workflow with explicit state management"""
    
    model = get_chat_model()
    prompt_registry = PromptRegistry()
    prompt = render_prompt(prompt_registry.get_prompt("simple_agent", "v2"), {})
    
    # Bind tools to the model
    doc_chunk_tool = create_doc_getter(doc_manager)
    schema_getter_tools = create_schema_getter()
    tools = [doc_chunk_tool, validate_ISA_schema] + schema_getter_tools
    
    model_with_tools = model.bind_tools(tools)
    
    # Define nodes
    def construction_node(state: ISAExtractionState) -> ISAExtractionState:
        """Agent constructs draft ISA metadata"""
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="Create ISA metadata for the provided document")]
        
        response = model_with_tools.invoke([SystemMessage(content=prompt.content)] + messages)
        
        return {
            "messages": messages + [response],
            "draft_isa_metadata": state.get("draft_isa_metadata")  # Preserve existing
        }
    
    def validation_node(state: ISAExtractionState) -> ISAExtractionState:
        """Validates the constructed ISA schema"""
        if not state.get("draft_isa_metadata"):
            return {"messages": state.get("messages", [])}
        
        # Call validation tool directly
        validation_result = validate_ISA_schema(state["draft_isa_metadata"])
        
        # Add validation result to messages
        messages = state.get("messages", [])
        messages.append(ToolMessage(
            content=str(validation_result),
            tool_call_id="validation"
        ))
        
        return {"messages": messages, "validation_result": validation_result}
    
    def tool_execution_node(state: ISAExtractionState) -> ISAExtractionState:
        """Executes tool calls from the model"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        
        # Execute each tool call
        tool_map = {
            "get_doc_chunk": doc_chunk_tool,
            "validate_ISA_schema": validate_ISA_schema,
            **{f"get_{t.name.lower()}_schema": t for t in schema_getter_tools}
        }
        
        for tool_call in last_message.tool_calls:
            tool = tool_map.get(tool_call["name"])
            if tool:
                result = tool.invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
                
                # If this is validation, store ISA in state
                if tool_call["name"] == "validate_ISA_schema":
                    state["draft_isa_metadata"] = tool_call["args"].get("isa")
        
        return {"messages": messages, "draft_isa_metadata": state.get("draft_isa_metadata")}
    
    def route_validation(state: ISAExtractionState) -> str:
        """Routes based on validation result"""
        validation_result = state.get("validation_result")
        if validation_result and isinstance(validation_result, dict):
            if validation_result.get("valid") is False:
                return "construct"
        return "end"
    
    # Build graph
    graph = StateGraph(ISAExtractionState)
    graph.add_node("construct", construction_node)
    graph.add_node("tools", tool_execution_node)
    graph.add_node("validate", validation_node)
    graph.add_node("end", lambda x: x)
    
    graph.add_edge(START, "construct")
    graph.add_conditional_edges(
        "construct",
        lambda s: "tools" if s["messages"][-1].tool_calls else "end"
    )
    graph.add_edge("tools", "validate")
    graph.add_conditional_edges("validate", route_validation)
    graph.add_edge("end", END)
    
    return graph.compile()


def visualize_graph(graph, output_path: str = "graph_visualization.png") -> str:
    """
    Visualizes a LangGraph graph and saves it as an image.
    
    Args:
        graph: The compiled LangGraph graph to visualize
        output_path: Path where the visualization should be saved (supports .png, .jpg, .svg)
    
    Returns:
        str: Path to the saved visualization file
    """
    try:
        # First, try to get the ASCII diagram to debug
        ascii_diagram = graph.get_graph().draw_mermaid()
        print("Generated Mermaid diagram:")
        print(ascii_diagram)
        print("\n" + "="*50 + "\n")
        
        # Try to generate PNG
        image_data = graph.get_graph().draw_mermaid_png()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "wb") as f:
            f.write(image_data)
        
        print(f"Graph visualization saved to: {output_file.absolute()}")
        return str(output_file.absolute())
        
    except Exception as e:
        print(f"Could not generate PNG visualization: {e}")
        print("Falling back to ASCII diagram...")
        
        try:
            # Get ASCII diagram for inspection
            ascii_diagram = graph.get_graph().draw_mermaid()
            
            output_file = Path(output_path.replace(".png", ".txt"))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w") as f:
                f.write(ascii_diagram)
            
            print(f"ASCII diagram saved to: {output_file.absolute()}")
            print("\nGraph structure:")
            print(ascii_diagram)
            
            # Try saving as SVG instead
            try:
                svg_data = graph.get_graph().draw_mermaid_svg()
                svg_file = Path(output_path.replace(".png", ".svg"))
                
                with open(svg_file, "wb") as f:
                    f.write(svg_data)
                
                print(f"\nAlso saved SVG version to: {svg_file.absolute()}")
                return str(svg_file.absolute())
            except Exception as svg_e:
                print(f"Could not generate SVG: {svg_e}")
                return str(output_file.absolute())
            
        except Exception as e2:
            print(f"Failed to generate ASCII diagram: {e2}")
            raise


