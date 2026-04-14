from pydantic import BaseModel, Field
from typing import List

from isatools.model import Source
from src.graphs.types.DTOs.MaterialAttributeValueDTO import MaterialAttributeValueDTO

class SourceDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Source name")
    characteristics: List[MaterialAttributeValueDTO] | None = Field(description="Characteristics of the source")

    def to_isatools_model(self) -> Source:
        source = Source(
            name=self.name
        )

        try:
            source.characteristics = [char.to_isatools_model() for char in self.characteristics] if self.characteristics else None  
        except Exception:
            source.characteristics = None

        return source

