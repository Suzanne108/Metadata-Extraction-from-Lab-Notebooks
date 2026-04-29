from src.graphs.workflows.stateful_graph_workflow import run_stateful_graph_workflow, create_stateful_graph_workflow
from src.graphs.workflows.isa_overview_workflow import run_isa_overview_workflow
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
    print(f"ISA overview written to: {path}")
    return path

if __name__ == "__main__":

    document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf" 

    run_stateful_graph_workflow(document_path=str(document_path))








    

