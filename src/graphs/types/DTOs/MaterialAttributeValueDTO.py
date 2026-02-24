from pydantic import BaseModel, Field

from isatools.model import Characteristic

from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO


class MaterialAttributeValueDTO(BaseModel):
    category: OntologyAnnotationDTO 
    value: str = Field(description="Value of the material attribute")
    unit: OntologyAnnotationDTO | None

    def to_isatools_model(self) -> Characteristic:
        return Characteristic(
            category=self.category.to_isatools_model(),
            value=self.value,
            unit=self.unit.to_isatools_model() if self.unit else None
        )