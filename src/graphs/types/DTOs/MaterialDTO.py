from pydantic import BaseModel, Field

from isatools.model import Material


class MaterialDTO(BaseModel):
    #TODO: the material object is not complete
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Material name")
    # = Field(description="Material type (e.g., Extract Name, Labeled Extract Name)")
    # characteristics: List[MaterialAttributeValue] | None = Field(description="Material characteristics")

    def to_isatools_model(self) -> Material:
        return Material(
            name=self.name,
        )
