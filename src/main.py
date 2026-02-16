from src.graphs.workflows.stateful_graph_workflow import run_stateful_graph_workflow, create_stateful_graph_workflow
from src.graphs.utils import visualise_graph
from pathlib import Path
from isatools.isajson import ISAJSONEncoder
import json
from isatools.model import Investigation, Study, Person, Publication

from src.state import ISAExtractionState



if __name__ == "__main__":

    #document_path = Path.cwd() / "papers" / "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf "

    #document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf"

    document_path = Path.cwd() / "papers" / "lab_journals" / "Dryzotron_protocol_testing.pdf"

    #visualise_graph(create_stateful_graph_workflow(document_path))

    run_stateful_graph_workflow(document_path=str(document_path))







    

