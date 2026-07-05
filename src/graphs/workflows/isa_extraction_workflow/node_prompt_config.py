from dataclasses import dataclass
from typing import Any

from src.graphs.types.GraphComponentId import GraphComponentId
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


@dataclass(frozen=True)
class SpecialistNodeConfig:
    node_id: GraphComponentId
    dto_cls: Any

    @property
    def base_name(self) -> str:
        return self.node_id.name.removesuffix("_CREATOR").lower()

    @property
    def prompt_key(self) -> str:
        return f"{self.base_name}_specialist"


SPECIALIST_NODE_CONFIGS: list[SpecialistNodeConfig] = [
    SpecialistNodeConfig(GraphComponentId.INVESTIGATION_CREATOR, InvestigationDTO),
    SpecialistNodeConfig(GraphComponentId.STUDY_CREATOR, StudyDTO),
    SpecialistNodeConfig(GraphComponentId.ASSAY_CREATOR, AssayDTO),
    SpecialistNodeConfig(GraphComponentId.PROCESS_CREATOR, ProcessDTO),
    SpecialistNodeConfig(GraphComponentId.PROTOCOL_CREATOR, ProtocolDTO),
    SpecialistNodeConfig(GraphComponentId.FACTOR_CREATOR, FactorDTO),
    SpecialistNodeConfig(GraphComponentId.SOURCE_CREATOR, SourceDTO),
    SpecialistNodeConfig(GraphComponentId.SAMPLE_CREATOR, SampleDTO),
    SpecialistNodeConfig(GraphComponentId.MATERIAL_CREATOR, MaterialDTO),
    SpecialistNodeConfig(GraphComponentId.DATAFILE_CREATOR, DataFileDTO),
]


def build_default_node_prompt_configs() -> dict[GraphComponentId, dict[str, str]]:
    """Build default per-node prompt config map for ISA extraction workflow."""
    default_configs: dict[GraphComponentId, dict[str, str]] = {
        GraphComponentId.DOC_OVERVIEW: {
            "prompt_key": "document_overview",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
        GraphComponentId.TASK_DECOMPOSER: {
            "prompt_key": "task_decomposer",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
        GraphComponentId.PLANNER_AGENT: {
            "prompt_key": "planner",
            "system_prompt_version": "v1-system",
            "policy_prompt_version": "v1-policy",
        },
        GraphComponentId.TEAM_LEAD: {
            "prompt_key": "isa_team_lead",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
        GraphComponentId.ASSEMBLY_ADVISOR: {
            "prompt_key": "assembly_advisor",
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        },
    }

    for specialist_config in SPECIALIST_NODE_CONFIGS:
        default_configs[specialist_config.node_id] = {
            "prompt_key": specialist_config.prompt_key,
            "system_prompt_version": "v1-system",
            "inference_prompt_version": "v1-inference",
        }

    return default_configs


DEFAULT_NODE_PROMPT_CONFIGS = build_default_node_prompt_configs()