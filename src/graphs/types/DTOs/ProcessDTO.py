from typing import Optional, List
from pydantic import BaseModel, Field

from isatools.model import Process
from src.graphs.types.DTOs.ProcessParameterValueDTO import ProcessParameterValueDTO


class ProcessDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Process name")
    performer: str = Field(description="Process performer")
    date: str = Field(description="Process date")
    executes_protocol: str | None = Field(description="Protocol executed by this process by id")
    parameter_values: List[ProcessParameterValueDTO] | None = Field(description="Parameter values used in the process")
    previous_processes: List[str] | None = Field(description="List of preceding processes by id")
    next_processes: List[str] | None = Field(description="List of succeeding processes by id")
    inputs: List[str] | None = Field(description="List of input materials or data files by id")
    outputs: List[str] | None = Field(description="List of output materials or data files by id")

    def to_isatools_model(self) -> Process:
        return Process(
            name=self.name,
            performer=self.performer,
            date_=self.date,
            parameter_values=[pv.to_isatools_model() for pv in self.parameter_values] if self.parameter_values else None
        )
