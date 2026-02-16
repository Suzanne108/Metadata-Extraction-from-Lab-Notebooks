import operator

from typing import TypedDict, Annotated, List, Dict

from langchain.messages import AnyMessage
from langchain_core.documents import Document
from src.graphs.types.NextStep import NextStep

from src.task import Task
from src.graphs.types.DTOs.InvestigationDTO import InvestigationDTO
from src.graphs.types.DTOs.PersonDTO import PersonDTO
from src.graphs.types.DTOs.PublicationDTO import PublicationDTO
from src.graphs.types.TeamLeadDecision import TeamLeadDecision
from src.graphs.types.DTOs.StudyDTO import StudyDTO
from src.graphs.types.DTOs.AssayDTO import AssayDTO
from src.graphs.types.DTOs.ProcessDTO import ProcessDTO
from src.graphs.types.DTOs.ProtocolDTO import ProtocolDTO
from src.graphs.types.DTOs.FactorDTO import FactorDTO
from src.graphs.types.DTOs.FactorValueDTO import FactorValueDTO
from src.graphs.types.DTOs.SourceDTO import SourceDTO
from src.graphs.types.DTOs.SampleDTO import SampleDTO
from src.graphs.types.DTOs.MaterialDTO import MaterialDTO
from src.graphs.types.DTOs.DataDTO import DataFileDTO


class ISAExtractionState(TypedDict):
    """Extended state for staged workflow: overview -> decomposition -> planner loop."""

    # Document to extract from
    document: Document | List[Document]

    # The latest validation results
    validation_result: dict | None

    # Count of consecutive validation failures and last failing path (for guardrails)
    consecutive_validation_failures: int | None
    last_validation_json_path: str | None

    # The latest decision from the orchestrator
    orchestrator_decision: NextStep | None

    # Evidence per query
    evidence: dict[str, list[str]]

    # The latest retrieved evidence
    latest_evidence: list[str] | None

    # Orientation: plain-text ISA mental model
    isa_mental_model: str | None

    # Overview: plain-text structured overview with numbered entity IDs
    document_overview: str | None

    # Backlog of tasks produced by the task decomposer
    tasks: List[Task] | None

    # Index of the task currently being worked on
    current_task_index: int | None

    # Message tracing
    messages: Annotated[list[AnyMessage], operator.add]

    # The latest executed node/action in the planner loop
    # Values: "SEARCH_DOCUMENT" | "UPDATE_DRAFT" | "VALIDATE" | "TASK_PROGRESS" | None
    last_action: str | None

    # Team lead routing decision object
    team_lead_decision: TeamLeadDecision | None
    
   
    # Registries of ISA objects by ID, always one investigation
    dto_investigation: InvestigationDTO | None
    registry_studies: Dict[str, StudyDTO] | None
    registry_assays: Dict[str, AssayDTO] | None
    registry_data: Dict[str, DataFileDTO] | None
    registry_processes: Dict[str, ProcessDTO] | None
    registry_protocols: Dict[str, ProtocolDTO] | None
    registry_factors: Dict[str, FactorDTO] | None
    registry_factor_values: Dict[str, FactorValueDTO] | None
    registry_sources: Dict[str, SourceDTO] | None
    registry_samples: Dict[str, SampleDTO] | None
    registry_materials: Dict[str, MaterialDTO] | None
