from src.graphs.workflows.stateful_graph_workflow import run_stateful_graph_workflow, create_stateful_graph_workflow
from src.graphs.utils import visualise_graph
from pathlib import Path
from isatools.isajson import ISAJSONEncoder
from isatools import isajson
from isatools.model import Investigation, Study, Person, Publication

from src.state import ISAExtractionState


def print_mock_arabidopsis_study_from_json() -> None:
    """Load the exported Mock Arabidopsis ISA-JSON and print the Study object.

    Assumes the file was written to output/Mock_Arabidopsis_Nitrogen_Study_extracted_isa.json
    using ISAJSONEncoder, and decodes it back into an Investigation with isajson.load.
    """

    json_path = Path.cwd() / "output" / "Mock_Arabidopsis_Nitrogen_Study_extracted_isa.json"
    if not json_path.exists():
        print(f"ISA-JSON file not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        inv = isajson.load(f)

    studies = getattr(inv, "studies", None) or []
    if not studies:
        print("No Study objects found in loaded Investigation.")
        return

    # Print the first study object; extend as needed for multiple studies
    first_study = studies[0]
    print("Loaded Study object from ISA-JSON:")
    print(first_study)


if __name__ == "__main__":

    #document_path = Path.cwd() / "papers" / "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf "

    #document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf"

    #document_path = Path.cwd() / "papers" / "lab_journals" / "Dryzotron_protocol_testing.pdf"

    #visualise_graph(create_stateful_graph_workflow(document_path))

    #run_stateful_graph_workflow(document_path=str(document_path))

    # After export, load the resulting ISA-JSON and print the Study object
    print_mock_arabidopsis_study_from_json()







    

