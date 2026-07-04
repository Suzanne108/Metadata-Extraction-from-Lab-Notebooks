from enum import Enum


class Action(Enum):
    SEARCH_DOCUMENT = "search contextual document"
    UPDATE_DRAFT = "ISA teamlead agent"
    FINISH_TASK = "task progress"
    STOP = "END"


class WorkflowAction(str, Enum):
    SEARCH_DOCUMENT = "SEARCH_DOCUMENT"
    UPDATE_DRAFT = "UPDATE_DRAFT"
    VALIDATE = "VALIDATE"
    TASK_PROGRESS = "TASK_PROGRESS"
    TEAM_LEAD_ROUTE = "TEAM_LEAD_ROUTE"
    ASSEMBLY_ADVICE = "ASSEMBLY_ADVICE"
    EXPORTED = "EXPORTED"

# LLM-visible actions exclude STOP and FINISH_TASK (termination is system-driven)
class ActionLLM(Enum):
    SEARCH_DOCUMENT = "search contextual document"
    UPDATE_DRAFT = "ISA drafting agent"

#TODO: convert to only using workflow action and Planner action, which are actions visible to the planner


#TODO: should this folder be named types or is there some other best practice for this?