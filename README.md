# Metadata Extraction from Lab Journals

A multi-agent LLM system that extracts structured metadata from scientific lab journal PDFs and converts it into [ISA (Investigation, Study, Assay)](https://isa-tools.org/) compliant JSON. Built with [LangGraph](https://langchain-ai.github.io/langgraph/).

This is the code for the master thesis `Metadata Extraction from Scientific Lab Notebooks: Design and Evaluation of a Mulit-Agent System.` by Suzanne Backer, Tu Delft 2026.

---

## Background

Lab journals contain rich experimental metadata — protocols, factors, sample descriptions, and measurements — but this information is typically written in free-form text. The ISA framework provides a standardised schema for representing this metadata, enabling FAIR (Findable, Accessible, Interoperable, Reusable) data practices. This project bridges the gap by proving a multi-agent system to extract and structure that information. 

## How It Works

The system processes a PDF lab journal through a stateful graph workflow:

1. **Document Overview** — The document is loaded and indexed into a vector store; an LLM produces a structured overview.
2. **Task Decomposition** — The overview is broken down into a list of ISA extraction tasks.
3. **Specialist Agents** — A planner routes tasks to dedicated agents for each ISA component: Investigation, Study, Assay, Source, Sample, Material, Protocol, Process, Factor, and DataFile.
4. **Assembly Advisor** — Reviews and validates the assembled ISA objects, flagging missing or inconsistent metadata.
5. **Export** — The final ISA object tree is written to a JSON file.

## Project Structure

```
src/
├── main.py                     # Single-run entry point that loads one experiment setup and runs it
├── model.py                    # LLM and embedding model configuration
├── state.py                    # LangGraph state definition
├── document_processing.py      # PDF loading and vector store indexing
├── task.py                     # Definition of Task object
├── tools.py                    # LangChain tool definitions
├── experiments/                # Declarative experiment configuration and runners
│   ├── experiment_setup.py     # ExperimentSetup dataclass (variant_name, model/prompt overrides)
│   ├── run_experiments.py      # Loader and batch runner for one or more setup files
│   ├── isa_model_comparison/   # Setup variants for model allocation experiments
│   └── planner_prompt_comparison/ # Setup variants for planner prompt version experiments
├── graphs/
│   ├── workflows/              # Full graph workflow definitions
│   ├── nodes/                  # Individual graph node implementations
│   └── types/                  # DTOs and type definitions for ISA objects
└── prompts/                    # YAML prompt templates per agent/version
instructions/
└── schemas/                    # JSON schemas for all ISA object types
output/                         # Extracted ISA JSON files and logs
papers/                         # An AI-generated mock lab notebook/paper
```

## Requirements

- Python 3.11 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)
    - Altneratively, other LLM providers can easily be integrated into the workflow by adapting `model.py`

## Installation

**1. Clone the repository**

```bash
git clone <repository-url>
cd metadata-extraction-from-lab-journals
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Run the extraction workflow with an experiment setup

The workflow is configured through an `ExperimentSetup` object loaded from a setup file in the `experiments` folder

```python
from pathlib import Path
from src.experiments.run_experiments import load_setup_from_file, run_experiment_variant

document_path = Path.cwd() / "papers" / "lab_journals" / "Example_journal.pdf"
setup = load_setup_from_file(
    Path(__file__).parent / "experiments" / "isa_model_comparison" / "setup_hybrid.py"
)

run_experiment_variant(setup=setup, document_path=document_path)
```

Run from the project root:

```bash
python -m src.main
```


### Define a new experiment setup

Create a new file in `src/experiments/...` that exposes an `EXPERIMENT_SETUP` object.

### Regenerate the ISA mental model

The ISA mental model is a plain-text explanation of the ISA framework used to guide the agents. To regenerate it:

```python
from src.main import generate_isa_mental_model
from pathlib import Path

generate_isa_mental_model(Path("instructions/isa_mental_model.txt"))
```

### Visualise the workflow graph

```python
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import ISAExtractionWorkflowBuilder
from src.graphs.utils import visualise_graph
from src.graphs.utils import index_document
from pathlib import Path

document_path = Path("papers/lab_journals/your_journal.pdf")
document_manager = index_document(str(document_path))
visualise_graph(ISAExtractionWorkflowBuilder(document_manager=document_manager).build())
```

## Output Format

Each run produces:
- `<journal_name>_extracted_isa.json` — the full ISA object tree (Investigation → Studies → Assays → Processes, Samples, Protocols, etc.)
- `<journal_name>_assembly_advice.txt` — the assembly advisor's advise on how extracted ISA entities should be connected to create a proper ISA workflow.
- `stream_log_<run_id>.txt` — full message trace for the run
- `isa_draft_<run_id>.json` — final in-memory `draft_isa_metadata` snapshot

