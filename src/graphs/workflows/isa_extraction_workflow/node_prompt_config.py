from typing import Any

from src.graphs.types.Node import GraphComponentId
from src.graphs.types.DTOs.InvestigationDTO import InvestigationDTO
from src.graphs.types.DTOs.StudyDTO import StudyDTO
from src.graphs.types.DTOs.AssayDTO import AssayDTO
from src.graphs.types.DTOs.ProcessDTO import ProcessDTO
from src.graphs.types.DTOs.ProtocolDTO import ProtocolDTO
from src.graphs.types.DTOs.FactorDTO import FactorDTO
from src.graphs.types.DTOs.SourceDTO import SourceDTO
from src.graphs.types.DTOs.SampleDTO import SampleDTO
from src.graphs.types.DTOs.MaterialDTO import MaterialDTO
from src.graphs.types.DTOs.DataDTO import DataFileDTO


SPECIALIST_NODE_CONFIGS: list[tuple[GraphComponentId, Any, str, str, str]] = [
    (GraphComponentId.INVESTIGATION_CREATOR, InvestigationDTO, "investigation_specialist", "dto_investigation", "INVESTIGATION_AGENT"),
    (GraphComponentId.STUDY_CREATOR, StudyDTO, "study_specialist", "dto_study", "STUDY_AGENT"),
    (GraphComponentId.ASSAY_CREATOR, AssayDTO, "assay_specialist", "dto_assay", "ASSAY_AGENT"),
    (GraphComponentId.PROCESS_CREATOR, ProcessDTO, "process_specialist", "dto_process", "PROCESS_AGENT"),
    (GraphComponentId.PROTOCOL_CREATOR, ProtocolDTO, "protocol_specialist", "dto_protocol", "PROTOCOL_AGENT"),
    (GraphComponentId.FACTOR_CREATOR, FactorDTO, "factor_specialist", "dto_factor", "FACTOR_AGENT"),
    (GraphComponentId.SOURCE_CREATOR, SourceDTO, "source_specialist", "dto_source", "SOURCE_AGENT"),
    (GraphComponentId.SAMPLE_CREATOR, SampleDTO, "sample_specialist", "dto_sample", "SAMPLE_AGENT"),
    (GraphComponentId.MATERIAL_CREATOR, MaterialDTO, "material_specialist", "dto_material", "MATERIAL_AGENT"),
    (GraphComponentId.DATAFILE_CREATOR, DataFileDTO, "datafile_specialist", "dto_data", "DATAFILE_AGENT"),
]


def build_default_node_prompt_configs() -> dict[str, dict[str, str]]:
    """Build default per-node prompt config map for ISA extraction workflow."""
    default_configs: dict[str, dict[str, str]] = {
        GraphComponentId.DOC_OVERVIEW.value: {
            "prompt_key": "document_overview",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
        GraphComponentId.TASK_DECOMPOSER.value: {
            "prompt_key": "task_decomposer",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
        GraphComponentId.PLANNER_AGENT.value: {
            "prompt_key": "planner",
            "system_prompt_version": "v1-system",
            "policy_prompt_version": "v1-policy",
        },
        GraphComponentId.TEAM_LEAD.value: {
            "prompt_key": "isa_team_lead",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
        GraphComponentId.ASSEMBLY_ADVISOR.value: {
            "prompt_key": "assembly_advisor",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
    }

    for specialist_node, _, specialist_prompt_key, _, _ in SPECIALIST_NODE_CONFIGS:
        default_configs[specialist_node.value] = {
            "prompt_key": specialist_prompt_key,
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        }

    return default_configs


DEFAULT_NODE_PROMPT_CONFIGS = build_default_node_prompt_configs()