from enum import Enum


class WorkflowAction(str, Enum):
    SEARCH_DOCUMENT = "SEARCH_DOCUMENT"
    UPDATE_DRAFT = "UPDATE_DRAFT"
    FINISH_TASK = "FINISH_TASK"
    STOP = "STOP"
    VALIDATE = "VALIDATE"
    TASK_PROGRESS = "TASK_PROGRESS"
    TEAM_LEAD_ROUTE = "TEAM_LEAD_ROUTE"
    ASSEMBLY_ADVICE = "ASSEMBLY_ADVICE"
    EXPORTED = "EXPORTED"

# Planner-visible actions exclude STOP and FINISH_TASK (termination is system-driven)
class ActionPlanner(str, Enum):
    SEARCH_DOCUMENT = "search contextual document"
    UPDATE_DRAFT = "ISA drafting agent"

#TODO: should this folder be named types or is there some other best practice for this?