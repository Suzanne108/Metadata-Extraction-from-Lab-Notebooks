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

    document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf" 
    document_manager = index_document(str(document_path))
    initial_state = create_initial_state(document_manager)
    app = ISAExtractionWorkflowBuilder(document_manager=document_manager).build()
    app, config, run_id = run_graph_workflow(
        app=app,
        initial_state=initial_state,
        run_prefix="isa-extract",
        stream=True,
    )
    handle_final_state(app, config, run_id)


   







    

