from typing import Optional, Literal
from pydantic import BaseModel, Field
from isatools.model import DataFile


DataTypeLiteral = Literal[
    "Raw Data File",
    "Derived Data File",
    "Image File",
    "Acquisition Parameter Data File",
    "Derived Spectral Data File",
    "Protein Assignment File",
    "Raw Spectral Data File",
    "Peptide Assignment File",
    "Array Data File",
    "Derived Array Data File",
    "Post Translational Modification Assignment File",
    "Derived Array Data Matrix File",
    "Free Induction Decay Data File",
    "Metabolite Assignment File",
    "Array Data Matrix File",
]


class DataFileDTO(BaseModel):
    uid: Optional[str] = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Data file name")
    type: DataTypeLiteral

    def to_isatools_model(self) -> DataFile:
        data = DataFile(
            filename=self.name
        )

        # `type` is a JSON-safe string literal, map it directly to label
        data.label = self.type if self.type is not None else None

        return data
