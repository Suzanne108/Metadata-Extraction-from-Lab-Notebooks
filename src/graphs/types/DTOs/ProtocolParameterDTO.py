from pydantic import BaseModel, Field

from isatools.model import ProtocolParameter
from typing import Optional

from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO


class ProtocolParameterDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    parameter_name: OntologyAnnotationDTO

    def to_isatools_model(self) -> ProtocolParameter:
        # Guard against invalid defaults (e.g., FieldInfo) and None
        try:
            param = self.parameter_name.to_isatools_model() if isinstance(self.parameter_name, OntologyAnnotationDTO) else None
        except Exception:
            param = None
        return ProtocolParameter(
            parameter_name=param,
        )
