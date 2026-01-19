

from enum import Enum


class Node(Enum):
    ISA_OVERVIEW = "ISA overview"
    DOC_OVERVIEW = "document overview"
    TASK_DECOMPOSER = "task planning"
    SEARCH_DOCUMENT = "search contextual document"
    TASK_PROGRESS = "task progress"
    VALIDATE = "validate current version"
    UPDATE_DRAFT_AGENT = "ISA drafting agent"
    PLANNER_AGENT = "planner agent"
    STOP = "END"