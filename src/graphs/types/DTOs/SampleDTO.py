from typing import Optional
from pydantic import BaseModel, Field

from isatools.model import Sample


class SampleDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Sample name")
    # characteristics: List[MaterialAttributeValue] | None = Field(description="Characteristics of the sample")
    factor_value: Optional[str] = Field(description="Factor value for the sample by id")
    derivesFrom: Optional[str] = Field(description="Source material from which the sample is derived by id")

    def to_isatools_model(self) -> Sample:
        return Sample(
            name=self.name
        )
