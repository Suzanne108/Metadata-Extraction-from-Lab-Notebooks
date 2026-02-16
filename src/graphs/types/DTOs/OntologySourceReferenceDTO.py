from pydantic import BaseModel, Field

from isatools.model import OntologySource

# This class is not yet used anywhere, but it may be useful in the future.

class OntologySourceReferenceDTO(BaseModel):
    name: str = Field(description="Ontology source name")
    description: str = Field(description="Ontology source description")
    file: str = Field(description="Ontology source file")
    version: str = Field(description="Ontology source version")

    def to_isatools_model(self) -> OntologySource:
        return OntologySource(
            name=self.name,
            description=self.description,
            file=self.file,
            version=self.version,
        )
