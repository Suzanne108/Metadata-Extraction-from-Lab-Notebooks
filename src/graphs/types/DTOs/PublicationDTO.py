from typing import List, Optional

from pydantic import BaseModel, Field


from isatools.model import Publication

from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO 

class PublicationDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    title: str = Field(description="Publication title")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    pubmed_id: Optional[str] = Field(None, description="PubMed identifier")
    authors: Optional[List[str]] = Field(None, description="List of author names")
    status: Optional[OntologyAnnotationDTO] 

    def to_isatools_model(self) -> Publication:
        publication = Publication(
            title=self.title,
            doi=self.doi,
            pubmed_id=self.pubmed_id,
            author_list=self.authors
        )

        try:
            publication.status = self.status.to_isatools_model() if self.status else None
        except Exception:
            publication.status = None
        return publication
            


