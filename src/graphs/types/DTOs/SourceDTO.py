from pydantic import BaseModel, Field

from isatools.model import Source


class SourceDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Source name")
    # characteristics: List[MaterialAttributeValue] | None = Field(description="Characteristics of the source")

    def to_isatools_model(self) -> Source:
        return Source(
            name=self.name
        )
