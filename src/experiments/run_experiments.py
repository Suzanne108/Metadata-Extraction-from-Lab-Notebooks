from __future__ import annotations

from pathlib import Path
import runpy

from src.experiments.experiment_setup import ExperimentSetup
from src.graphs.workflows.isa_extraction_workflow.final_state import handle_final_state
from src.graphs.workflows.isa_extraction_workflow.initial_state import create_initial_state
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import (
    ISAExtractionWorkflowBuilder,
)
from src.graphs.workflows.workflow_runner import run_graph_workflow
from src.graphs.utils import index_document


def _load_setup_from_file(setup_file: Path) -> ExperimentSetup:
    """Load EXPERIMENT_SETUP from a setup file without hard package imports."""
    loaded = runpy.run_path(str(setup_file))
    setup = loaded["EXPERIMENT_SETUP"]
    return setup


def run_experiment_variant(setup: ExperimentSetup, document_path: Path) -> str:
    """Run one extraction experiment from a declarative setup."""
    document_manager = index_document(str(document_path))
    initial_state = create_initial_state(document_manager)
    app = ISAExtractionWorkflowBuilder(
        document_manager=document_manager,
        node_models=setup.node_models,
        default_model=setup.default_model,
        node_prompt_configs=setup.node_prompt_configs,
    ).build()

    app, config, run_id = run_graph_workflow(
        app=app,
        initial_state=initial_state,
        run_prefix=f"isa-extract-{setup.variant_name}",
        stream=False,
    )
    handle_final_state(app, config, run_id)
    return str(run_id)


def collect_experiments() -> list[ExperimentSetup]:
    """Collect experiment setups to run in order."""
    model_comparison_dir = Path(__file__).parent / "isa_model_comparison"
    planner_prompt_comparison_dir = Path(__file__).parent / "planner_prompt_comparison"

    return [
        _load_setup_from_file(model_comparison_dir / "setup_all_mini.py"),
        _load_setup_from_file(model_comparison_dir / "setup_all_nano.py"),
        _load_setup_from_file(model_comparison_dir / "setup_hybrid.py"),
        _load_setup_from_file(planner_prompt_comparison_dir / "setup_planner_system_v1.py"),
        _load_setup_from_file(planner_prompt_comparison_dir / "setup_planner_system_v2.py"),
    ]


def main() -> None:
    """Run all configured ISA extraction comparison experiments."""
    document_path = Path.cwd() / "papers" / "Mock_Arabidopsis_Nitrogen_Study.pdf"

    for setup in collect_experiments():
        run_id = run_experiment_variant(setup=setup, document_path=document_path)
        print(f"[{setup.variant_name}] run_id={run_id}")
        print(f"[{setup.variant_name}] setup={setup.description}")


if __name__ == "__main__":
    main()
