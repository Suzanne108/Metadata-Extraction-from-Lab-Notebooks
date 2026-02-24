from typing import List, Optional
from pydantic import BaseModel, Field

from isatools.model import Assay

from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO

#TODO: implement more objects to be recorded in the Assay type, otherwise score is low.
#TODO: maybe remove the data agent and instead directly produce the data files in the assay object


class AssayDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    technology_platform: str = Field(description="Technology platform used in the assay")
    measurement_type: OntologyAnnotationDTO | None 
    technology_type: OntologyAnnotationDTO | None 
    data_files: List[str] | None = Field(default=None, description="Data files associated with the assay by id")
    samples: List[str] | None = Field(default=None, description="Samples used in the assay by id")
    other_materials: List[str] | None = Field(default=None, description="Materials used in the assay by id")
    characteristics: List[OntologyAnnotationDTO] | None = Field(
        default=None, description="Characteristics of the assay (material attributes)")
    unit_categories: List[OntologyAnnotationDTO] | None = Field(
        default=None, description="Unit categories used in the assay (ontology annotations)")
    process_sequence: List[str] | None = Field(
        default=None, description="Process sequence associated with the assay by id")
    

    def to_isatools_model(self) -> Assay:
        assay = Assay(
            technology_platform=self.technology_platform,
            measurement_type=self.measurement_type.to_isatools_model() if self.measurement_type else None,
            technology_type=self.technology_type.to_isatools_model() if self.technology_type else None,
        )

        # Only set attributes if present on DTO and Assay
        try:
            if hasattr(self, "unit_categories") and self.unit_categories:
                assay.units = [uc.to_isatools_model() for uc in self.unit_categories]
        except Exception:
            pass

        # Avoid referencing undefined `characteristics` on DTO
        try:
            if hasattr(self, "characteristics") and self.characteristics:
                assay.characteristic_categories = [attr.to_isatools_model() for attr in self.characteristics]
        except Exception:
            pass

        return assay
