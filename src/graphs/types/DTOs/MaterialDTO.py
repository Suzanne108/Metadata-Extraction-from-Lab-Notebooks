from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from isatools.model import Material

from src.graphs.types.DTOs.MaterialAttributeValueDTO import MaterialAttributeValueDTO


MaterialTypeLiteral = Literal[
    "Extract Name",
    "Labeled Extract Name",
]


class MaterialDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Material name")
    type: Optional[MaterialTypeLiteral] = Field(description="Material type (e.g., Extract Name, Labeled Extract Name)")
    characteristics: List[MaterialAttributeValueDTO] | None = Field(description="Material characteristics")

    def to_isatools_model(self) -> Material:

        material = Material(
            name=self.name,
            type_=self.type if self.type else None,
            characteristics=[char.to_isatools_model() for char in self.characteristics] if self.characteristics else None
        )
        
        return material
    
