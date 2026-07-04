from src.graphs.workflows.workflow_runner import run_graph_workflow
from src.graphs.workflows.isa_extraction_workflow.initial_state import create_initial_state
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import create_stateful_graph_workflow
from src.graphs.workflows.isa_extraction_workflow.final_state import handle_final_state
from src.graphs.workflows.isa_overview_workflow.isa_overview_workflow import run_isa_overview_workflow
from src.graphs.utils import visualise_graph
from pathlib import Path

from src.plotting_results.category_assessment_plots import (
    plot_results_grouped,
    plot_results_stacked,
)



def generate_isa_mental_model(output_path: Path | None = None) -> Path:
    """Run the ISA-overview-only workflow and return the output file path.

    This is a convenience wrapper so other code (or interactive sessions) can
    trigger regeneration of the fixed ISA explanation used by the main graph.
    """

    path = run_isa_overview_workflow(output_path)
    #TODO: remove unnecessary print statemtents
    print(f"ISA overview written to: {path}")
    return path

if __name__ == "__main__":

    document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf" 

    #TODO: change that document_path should only be injected once
    #TODO: add options to tweak the document embedding strategy

    initial_state = create_initial_state(str(document_path))
    app = create_stateful_graph_workflow(str(document_path))
    app, config, run_id = run_graph_workflow(
        app=app,
        initial_state=initial_state,
        run_prefix="isa-extract",
        stream=True,
    )
    handle_final_state(app, config, run_id)


    #TODO:add tests to for example test the hardcoded parts of the workflow.

    #TODO: use the experiments runner







    

