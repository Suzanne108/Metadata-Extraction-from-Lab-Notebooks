from pydantic import BaseModel, Field

from isatools.model import Investigation
from typing import List
from src.graphs.types.DTOs.PersonDTO import PersonDTO
from src.graphs.types.DTOs.PublicationDTO import PublicationDTO


class InvestigationDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    identifier: str = Field(description="The unique identifier for the investigation")
    title: str = Field(description="The title of the investigation")
    description: str = Field(description="A brief description of the investigation")
    submission_date: str = Field(description="The date the investigation was submitted")
    public_release_date: str = Field(description="The date the investigation was made publicly available")
    #ontology_source_references: List[OntologySourceReference] | None = Field(description="A list of ontology source references used in the investigation")
    publications: List[PublicationDTO] | None = Field(description="A list of publications associated with the investigation")
    contacts: List[PersonDTO] | None = Field(description="A list of contacts for the investigation")

    def to_isatools_model(self) -> Investigation:
        investigation = Investigation(
            identifier=self.identifier,
            title=self.title,
            description=self.description,
            submission_date=self.submission_date,
            public_release_date=self.public_release_date
        )

        # Convert nested DTOs to isatools models
        try:
            if self.publications is not None and hasattr(investigation, "publications"):
                investigation.publications = [p.to_isatools_model() for p in self.publications]
        except Exception:
            pass
        try:
            if self.contacts is not None and hasattr(investigation, "contacts"):
                investigation.contacts = [c.to_isatools_model() for c in self.contacts]
        except Exception:
            pass
        
        return investigation