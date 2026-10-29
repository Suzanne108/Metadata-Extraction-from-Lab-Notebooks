from pydantic import BaseModel, Field

from isatools.model import StudyFactor
from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO


class FactorDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    factor_name: str = Field(description="Factor name")
    factor_type: OntologyAnnotationDTO | None 

    def to_isatools_model(self) -> StudyFactor:
        return StudyFactor(
            name=self.factor_name,
            factor_type=self.factor_type.to_isatools_model() if self.factor_type else None
        )
