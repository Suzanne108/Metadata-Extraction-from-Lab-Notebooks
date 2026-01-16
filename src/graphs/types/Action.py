from enum import Enum


class Action(Enum):
    SEARCH_DOCUMENT = "search contextual document"
    VALIDATE = "validate current version"
    UPDATE_DRAFT = "ISA drafting agent"
    FINISH_TASK = "task progress"
    STOP = "END"

# LLM-visible actions exclude STOP and FINISH_TASK (termination is system-driven)
class ActionLLM(Enum):
    SEARCH_DOCUMENT = "search contextual document"
    VALIDATE = "validate current version"
    UPDATE_DRAFT = "ISA drafting agent"