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

        # isatools enforces that if a unit is supplied, the value must be
        # quantitative (numeric). To avoid runtime errors when the LLM produces
        # free-text values with units, attempt to coerce to float; if that
        # fails, drop the unit so the value is treated as qualitative.
        value: str | float | None = self.value_text
        if unit is not None and self.value_text is not None:
            try:
                value = float(self.value_text)
            except (TypeError, ValueError):
                # Non-numeric value with a unit: treat as qualitative by
                # removing the unit to satisfy isatools' constraints.
                unit = None

        return ParameterValue(
            category=category,
            value=value,
            unit=unit,
        )
