from typing import Optional
from pydantic import BaseModel, Field

from isatools.model import FactorValue
from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO


class FactorValueDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    category_name: str = Field(description="Factor category name by id")
    value_text: str = Field(description="Factor value as text")
    unit_text: OntologyAnnotationDTO | None

    def to_isatools_model(self) -> FactorValue:
        factor_value = FactorValue(
            factor_name=self.category_name,
            value=self.value_text,
            unit= self.unit_text.to_isatools_model() if isinstance(self.unit_text, OntologyAnnotationDTO) else None
        )


        return factor_value
