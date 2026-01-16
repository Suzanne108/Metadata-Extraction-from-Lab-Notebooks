from typing import Literal
from pydantic import BaseModel

class Task(BaseModel):
    kind: Literal["investigation", "study", "assay", "material", "process", "protocol", "factor", "data"]
    target: str
    instruction: str
    status: Literal["pending", "in-progress", "completed"]

class TaskList(BaseModel):
    tasks: list[Task]