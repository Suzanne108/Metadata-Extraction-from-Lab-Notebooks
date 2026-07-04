from typing import Literal
from pydantic import BaseModel

#TODO: should there not be some kind of general ENUM in the codebase for the different specialist agent types? This would be more maintainable than hardcoding the strings in multiple places.

class Task(BaseModel):
    kind: Literal["investigation", "study", "assay", "material", "process", "protocol", "factor", "datafile", "source", "sample"]
    target: str
    instruction: str
    status: Literal["pending", "in-progress", "completed"]

class TaskList(BaseModel):
    tasks: list[Task]

    #TODO: Not sure this file should be here since it is very specific to the ISA extraction workflow. Should it be moved to a more specific location in the codebase?