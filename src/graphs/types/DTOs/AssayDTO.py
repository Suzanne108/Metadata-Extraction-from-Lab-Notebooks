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
    process_sequence: List[str] | None = Field(
        default=None, description="Process sequence associated with the assay by id")
    

    def to_isatools_model(self) -> Assay:
        assay = Assay(
            technology_platform=self.technology_platform,
            measurement_type=self.measurement_type.to_isatools_model() if self.measurement_type else None,
            technology_type=self.technology_type.to_isatools_model() if self.technology_type else None,
        )

        return assay
