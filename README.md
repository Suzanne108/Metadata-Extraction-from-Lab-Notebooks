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
├── main.py                     # Entry point
├── model.py                    # LLM and embedding model configuration
├── state.py                    # LangGraph state definition
├── document_processing.py      # PDF loading and vector store indexing
├── task.py                     # Definition of Task object
├── tools.py                    # LangChain tool definitions
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

### Extract ISA metadata from a lab journal

Edit `src/main.py` and set the path to your PDF lab journal, then run:

```python
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import ISAExtractionWorkflowBuilder
from src.graphs.workflows.isa_extraction_workflow.initial_state import create_initial_state
from src.graphs.workflows.workflow_runner import run_graph_workflow
from pathlib import Path

document_path = Path("papers/your_journal.pdf")
app = ISAExtractionWorkflowBuilder(document_path=str(document_path)).build()
initial_state = create_initial_state(str(document_path))
run_graph_workflow(app=app, initial_state=initial_state)
```

Run from the project root:

```bash
python -m src.main
```

The extracted ISA JSON and a run log will be written to the `output/` directory.

### Regenerate the ISA mental model

The ISA mental model is a plain-text explanation of the ISA framework used to guide the agents. To regenerate it:

```python
from src.main import generate_isa_mental_model
from pathlib import Path

generate_isa_mental_model(Path("output/ISA_mental_model.txt"))
```

### Visualise the workflow graph

```python
from src.graphs.workflows.isa_extraction_workflow.isa_extraction_workflow_builder import ISAExtractionWorkflowBuilder
from src.graphs.utils import visualise_graph
from pathlib import Path

document_path = Path("papers/lab_journals/your_journal.pdf")
visualise_graph(ISAExtractionWorkflowBuilder(document_path=str(document_path)).build())
```

## Output Format

Each run produces:
- `<journal_name>_extracted_isa.json` — the full ISA object tree (Investigation → Studies → Assays → Processes, Samples, Protocols, etc.)
- `<journal_name>_assembly_advice.txt` — the assembly advisor's advise on how extracted ISA entities should be connected to create a proper ISA workflow.
- `stream_log_<run_id>.txt` — full message trace for the run

