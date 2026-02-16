from typing import Optional, List
from pydantic import BaseModel, Field

from isatools.model import Study   
from src.graphs.types.DTOs.PersonDTO import PersonDTO
from src.graphs.types.DTOs.PublicationDTO import PublicationDTO


class StudyDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    identifier: str = Field(description="The unique identifier for the study")
    title: str = Field(description="The title of the study")
    description: str = Field(description="A brief description of the study")
    submission_date: str = Field(description="The date the study was submitted")
    public_release_date: str = Field(description="The date the study was made publicly available")
    publications: List[PublicationDTO] | None = Field(description="A list of publications associated with the study")
    people: List[PersonDTO] | None = Field(description="A list of contacts for the study")
    protocols: List[str] | None = Field(description="A list of protocols in the study")
    assays: List[str] | None = Field(description="A list of assays in the study")
    sources: List[str] | None = Field(description="A list of sources in the study")
    samples: List[str] | None = Field(description="A list of samples in the study")
    other_materials: List[str] | None = Field(description="A list of other materials in the study")
    process_sequence: List[str] | None = Field(description="The process sequence of the study")
    factors: List[str] | None = Field(description="Experimental factors in the study")

    def to_isatools_model(self) -> Study:
        study =  Study(
            identifier=self.identifier,
            title=self.title,
            description=self.description,
            submission_date=self.submission_date,
            public_release_date=self.public_release_date,
        )

        if self.publications:
            study.publications = [pub.to_isatools_model() for pub in self.publications]
        if self.people:
            study.contacts = [person.to_isatools_model() for person in self.people]
        return study
    
