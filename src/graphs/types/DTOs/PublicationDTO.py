from typing import List, Optional

from pydantic import BaseModel, Field


from isatools.model import Publication 

class PublicationDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    title: str = Field(description="Publication title")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    pubmed_id: Optional[str] = Field(None, description="PubMed identifier")
    authors: Optional[List[str]] = Field(None, description="List of author names")

    def to_isatools_model(self) -> Publication:
        return Publication(
            title=self.title,
            doi=self.doi,
            pubmed_id=self.pubmed_id,
            author_list=self.authors
        )
       



