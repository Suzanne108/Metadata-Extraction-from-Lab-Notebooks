from src.experiments.run_experiments import load_setup_from_file, run_experiment_variant
from src.graphs.workflows.workflow_runner import run_graph_workflow
from src.graphs.workflows.isa_extraction_workflow.initial_state import create_initial_state
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import ISAExtractionWorkflowBuilder
from src.graphs.workflows.isa_extraction_workflow.final_state import handle_final_state
from src.graphs.workflows.isa_overview_workflow.isa_overview_workflow import run_isa_overview_workflow
from src.graphs.utils import index_document
from pathlib import Path



def generate_isa_mental_model(output_path: Path | None = None) -> Path:
    """Run the ISA-overview-only workflow and return the output file path.

    This is a convenience wrapper so other code (or interactive sessions) can
    trigger regeneration of the fixed ISA explanation used by the main graph.
    """

    path = run_isa_overview_workflow(output_path)
    print(f"ISA overview written to: {path}")
    return path

if __name__ == "__main__":

    document_path = Path.cwd() / "papers" / "lab_journals" / "N3.pdf"

    setup = load_setup_from_file(Path(__file__).parent / "experiments" / "isa_model_comparison" / "setup_hybrid.py")

    run_experiment_variant(setup=setup, document_path=document_path)
    
    

    


   







    

