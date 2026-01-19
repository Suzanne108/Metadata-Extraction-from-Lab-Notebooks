from dataclasses import Field

from pydantic import BaseModel, Field

from src.graphs.types.Action import Action, ActionLLM


class NextStep(BaseModel):
    action: Action = Field(description= "The next step to perform")
    instruction: str | None = Field(description= "In case of action update_draft, the instruction to be given to the ISA updater agent")
    query: str | None = Field(description= "In case of action search_document, the query to search the document with")
    k: int | None = Field(description= "In case of search_document, the amount of semantically relevant pieces to retrieve")
    rationale: str = Field(description= "The reason for choosing this action")

class NextStepLLM(BaseModel):
    action: ActionLLM = Field(description="The next step to perform (LLM-restricted)")
    instruction: str | None = Field(description="In case of action update_draft, the instruction to be given to the ISA updater agent")
    query: str | None = Field(description="In case of action search_document, the query to search the document with")
    k: int | None = Field(description="In case of search_document, the amount of semantically relevant pieces to retrieve")
    rationale: str = Field(description="The reason for choosing this action")