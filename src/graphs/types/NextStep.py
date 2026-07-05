from pydantic import BaseModel, Field

from src.graphs.types.Action import WorkflowAction, ActionPlanner


class NextStep(BaseModel):
    action: WorkflowAction = Field(description= "The next step to perform")
    instruction: str | None = Field(description= "In case of action update_draft, the instruction to be given to the ISA updater agent")
    query: str | None = Field(description= "In case of action search_document, the query to search the document with")
    k: int | None = Field(description= "In case of search_document, the amount of semantically relevant pieces to retrieve")
    rationale: str = Field(description= "The reason for choosing this action")

class NextStepLLM(BaseModel):
    action: ActionPlanner = Field(description="The next step to perform (planner-restricted)")
    instruction: str | None = Field(description="In case of action update_draft, the instruction to be given to the ISA updater agent")
    query: str | None = Field(description="In case of action search_document, the query to search the document with")
    k: int | None = Field(description="In case of search_document, the amount of semantically relevant pieces to retrieve")
    rationale: str = Field(description="The reason for choosing this action")