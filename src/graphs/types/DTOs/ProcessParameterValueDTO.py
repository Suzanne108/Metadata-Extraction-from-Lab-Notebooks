from pydantic import BaseModel, Field

from isatools.model import ParameterValue

from src.graphs.types.DTOs.ProtocolParameterDTO import ProtocolParameterDTO
from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO


class ProcessParameterValueDTO(BaseModel):
    category_name: ProtocolParameterDTO | None 
    value_text: str | None = Field(default=None, description="Parameter value as text")
    unit_text: OntologyAnnotationDTO | None 

    def to_isatools_model(self) -> ParameterValue:
        # Safely convert nested DTOs if present and of expected types
        try:
            category = self.category_name.to_isatools_model() if isinstance(self.category_name, ProtocolParameterDTO) else None
        except Exception:
            category = None
        try:
            unit = self.unit_text.to_isatools_model() if isinstance(self.unit_text, OntologyAnnotationDTO) else None
        except Exception:
            unit = None
        return ParameterValue(
            category=category,
            value=self.value_text,
            unit=unit,
        )
