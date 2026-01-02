from src.graphs.workflows.stateful_graph_workflow import run_stateful_graph_workflow, create_stateful_graph_workflow
from src.graphs.workflows.standalone_agent_workflow import standalone_agent_workflow
from src.document_processing import DocumentManager
from src.graphs.utils import visualise_graph
from pathlib import Path

# Example usage:
if __name__ == "__main__":

    #visualise_graph(create_stateful_graph_workflow())

    run_stateful_graph_workflow()




    

