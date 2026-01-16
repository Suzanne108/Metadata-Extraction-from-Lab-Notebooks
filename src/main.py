from src.graphs.workflows.stateful_graph_workflow import run_stateful_graph_workflow, create_stateful_graph_workflow
from src.graphs.workflows.standalone_agent_workflow import standalone_agent_workflow
from src.document_processing import DocumentManager
from src.graphs.utils import visualise_graph
from pathlib import Path

from src.state.v3 import ISAExtractionStateV3

# Example usage:
if __name__ == "__main__":

    document_path = Path.cwd() / "papers" / "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf "

    document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf"

    #visualise_graph(create_stateful_graph_workflow())

    final_state : ISAExtractionStateV3 = run_stateful_graph_workflow(document_path=str(document_path))

    print("Final extracted state:", final_state.draft_isa_metadata)






    

