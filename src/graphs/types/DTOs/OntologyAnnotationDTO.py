from pydantic import BaseModel, Field

from isatools.model import OntologyAnnotation


class OntologyAnnotationDTO(BaseModel):
    #TODO: expand ontology annotation fields
    annotation_value: str = Field(description="Annotation value as text or number string")
    #term_source: str = Field(description="Abbreviated ontology name")
    #term_accession: str = Field(description="Ontology term accession URI")

    def to_isatools_model(self) -> OntologyAnnotation:
        return OntologyAnnotation(
            term=self.annotation_value
        )
