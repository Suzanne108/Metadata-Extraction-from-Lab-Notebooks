from pydantic import BaseModel, Field

from isatools.model import Protocol
from src.graphs.types.DTOs.OntologyAnnotationDTO import OntologyAnnotationDTO
from src.graphs.types.DTOs.ProtocolParameterDTO import ProtocolParameterDTO

from typing import List


class ProtocolDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Protocol name")
    type: OntologyAnnotationDTO | None 
    description: str = Field(description="Protocol description")
    uri: str | None = Field(description="Protocol URI")
    version: str | None = Field(description="Protocol version")
    parameters: List[ProtocolParameterDTO] | None 

    def to_isatools_model(self) -> Protocol:
        proto = Protocol(
            name=self.name,
            description=self.description,
            uri=self.uri,
            version=self.version,
        )
        # Map ontology type if available (isatools uses `protocol_type` attribute)
        try:
            if self.type is not None and hasattr(proto, "protocol_type"):
                proto.protocol_type = self.type.to_isatools_model()
        except Exception:
            pass
        # Attach protocol parameters
        try:
            if self.parameters and hasattr(proto, "parameters"):
                proto.parameters = [p.to_isatools_model() for p in self.parameters]
        except Exception:
            pass
        return proto
