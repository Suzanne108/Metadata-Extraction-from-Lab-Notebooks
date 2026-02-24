

from enum import Enum


class Node(Enum):
    ISA_OVERVIEW = "ISA overview"
    DOC_OVERVIEW = "document overview"
    TASK_DECOMPOSER = "task planning"
    SEARCH_DOCUMENT = "search contextual document"
    TASK_PROGRESS = "task progress"
    VALIDATE = "validate current version"
    UPDATE_DRAFT_AGENT = "ISA drafting agent"
    TEAM_LEAD = "ISA team lead"
    INVESTIGATION_CREATOR = "investigation creator"
    STUDY_CREATOR = "study creator"
    ASSAY_CREATOR = "assay creator"
    PROCESS_CREATOR = "process creator"
    PROTOCOL_CREATOR = "protocol creator"
    FACTOR_CREATOR = "factor creator"
    SOURCE_CREATOR = "source creator"
    SAMPLE_CREATOR = "sample creator"
    MATERIAL_CREATOR = "material creator"
    DATAFILE_CREATOR = "datafile creator"
    ASSEMBLY_ADVISOR = "assembly advisor"
    ASSEMBLY = "deterministic assembly"
    PLANNER_AGENT = "planner agent"
    EXPORT = "investigation export"
    STOP = "END"