from typing import Optional, List
from pydantic import BaseModel, Field

from isatools.model import Sample

from src.graphs.types.DTOs.FactorValueDTO import FactorValueDTO

from src.graphs.types.DTOs.MaterialAttributeValueDTO import MaterialAttributeValueDTO

#TODO: implement isatools model conversion for factor values


class SampleDTO(BaseModel):
    uid: str = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Sample name")
    characteristics: List[MaterialAttributeValueDTO] | None = Field(description="Characteristics of the sample")
    factor_value: List[FactorValueDTO] | None = Field(description="Factor value for the sample by id")

    def to_isatools_model(self) -> Sample:
        sample = Sample(
            name=self.name
        )

        try:
            sample.characteristics = [char.to_isatools_model() for char in self.characteristics] if self.characteristics else None
        except Exception:
            pass

        try:
            sample.factor_values = [fv.to_isatools_model() for fv in self.factor_value] if self.factor_value else None
        except Exception:
            pass

        return sample
