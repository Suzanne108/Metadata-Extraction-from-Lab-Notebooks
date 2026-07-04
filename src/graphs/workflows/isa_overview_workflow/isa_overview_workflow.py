from pathlib import Path

from langgraph.graph import StateGraph

from src.graphs.core import WorkflowGraphBuilder
from src.state import ISAExtractionState
from src.graphs.types.Node import GraphComponentId
from src.graphs.nodes.LLM.isa_overview import ISAOverviewNode
from src.tools import create_schema_getter
from src.prompts.registry import PromptRegistry
from src.prompts.renderer import render_prompt
from src.model import get_chat_model
from src.document_processing import DocumentManager


def build_isa_overview_node(registry: PromptRegistry, schema_getter_tools):
    """Create the ISA overview node with its prompts.

    This is shared between the dedicated ISA-overview-only workflow and any
    other place that needs to construct the same node.
    """

    model = get_chat_model().bind_tools(schema_getter_tools)
    isa_overview_system = render_prompt(registry.get_prompt("ISA_overview", "v2-system"), {})
    isa_overview_infer = registry.get_prompt("ISA_overview", "v2-inference")

    doc_manager = DocumentManager()
    path = Path.cwd() / "instructions" / "ISA_abstract_model.pdf"

    isa_overview_doc = doc_manager.load_pdf(path)

    return ISAOverviewNode(
        model=model,
        schema_getter_tools=schema_getter_tools,
        system_prompt=isa_overview_system,
        inference_prompt_template=isa_overview_infer,
        isa_overview_doc=isa_overview_doc,
    )


def create_isa_overview_graph() -> StateGraph:
    """Create a minimal graph that only runs the ISA overview node.

    This graph is intended to be run separately to materialise a stable
    "mental model" description of ISA that can be reused across runs of the
    main extraction workflow.
    """

    # Reuse the same node construction as the main workflow used to do
    schema_getter_tools = create_schema_getter()
    registry = PromptRegistry.get_instance()
    isa_overview_node = build_isa_overview_node(registry, schema_getter_tools)

    nodes = {GraphComponentId.ISA_OVERVIEW.value: isa_overview_node}
    edges = [
        ("START", GraphComponentId.ISA_OVERVIEW.value),
        (GraphComponentId.ISA_OVERVIEW.value, "END"),
    ]

    # No checkpointer needed here; we only care about the final state.
    builder = WorkflowGraphBuilder(ISAExtractionState)
    return builder.add_nodes(nodes).add_edges(edges).compile()


# TODO: refactor this into a more general-purpose workflow runner that can be used for any graph, not just the ISA overview one.


def run_isa_overview_workflow(output_path: Path | None = None) -> Path:
    """Run the ISA overview graph once and persist its text to a file.

    The resulting file is intended to be used as a fixed input for the
    document overview node in the main workflow so that all extraction runs
    share the same ISA explanation.
    """

    app = create_isa_overview_graph()

    # Minimal initial state – the ISA overview node does not depend on the
    # input document, only on the ISA abstract model PDF it loads internally.
    initial_state: ISAExtractionState = {
        "document": None,
        "orchestrator_decision": None,
        "isa_mental_model": "",
        "document_overview": "",
        "tasks": [],
        "current_task_index": None,
        "messages": [],
        "last_action": None,
        "team_lead_decision": None,
        "dto_investigation": None,
        "registry_studies": None,
        "registry_assays": None,
        "registry_data": None,
        "registry_processes": None,
        "registry_protocols": None,
        "registry_factors": None,
        "registry_factor_values": None,
        "registry_sources": None,
        "registry_samples": None,
        "registry_materials": None,
    }

    final_state = app.invoke(initial_state)
    mental_model = final_state.get("isa_mental_model", "") or ""

    if output_path is None:
        output_path = Path.cwd() / "instructions" / "isa_mental_model.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mental_model, encoding="utf-8")

    return output_path


