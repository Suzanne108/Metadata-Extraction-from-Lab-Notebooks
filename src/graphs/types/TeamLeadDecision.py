from typing import Literal

from pydantic import BaseModel, Field


class TeamLeadDecision(BaseModel):
    """Routing decision produced by the ISA team lead.

    - target: which specialized agent should handle the subtask
    - instruction: actionable instruction forwarded to that agent
    - rationale: brief explanation of the routing choice
    """

    target: Literal[
        "investigation",
        "study",
        "assay",
        "process",
        "protocol",
        "factor",
        "source",
        "sample",
        "material",
        "datafile"
    ] = Field(...)
    instruction: str = Field(...)
    rationale: str = Field(...)


