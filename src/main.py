from src.graphs.workflows.simple_workflow_v3 import create_simple_workflow_v3, visualize_graph
from src.graphs.workflows.standalone_agent_workflow import standalone_agent_workflow
from src.document_processing import DocumentManager
from pathlib import Path

# Example usage:
if __name__ == "__main__":


    standalone_agent_workflow()



    # Create document manager and workflow
    #doc_manager = DocumentManager()
    #graph = create_simple_workflow_v3(doc_manager)
    
    # Visualize the graph
    #visualize_graph(graph, "output/workflow_visualization.png")
    

