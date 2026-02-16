from typing import Optional, Literal
from pydantic import BaseModel, Field

from isatools.model import DataFile


class DataFileDTO(BaseModel):
    uid: Optional[str] = Field(default=None, description="Stable identifier for reuse/reference")
    name: str = Field(description="Data file name")
    #TODO fix, this should be an enum
    type: Literal["Raw Data File", 
                  "Derived Data File", 
                  "Image File", 
                  "Acquisition Parameter Data File",
                  "Derived Spectral Data File," \
                  "Protein Assignment File",
                  "Raw Spectral Data File",
                  "Peptide Assignment File",
                  "Array Data File",
                  "Derived Array Data File",
                  "Post Translational Modification Assignment File",
                  "Derived Array Data Matrix File",
                  "Free Induction Decay Data File",
                  "Metabolite Assignment File",
                  "Array Data Matrix File"] = Field(description="Data file type")

    def to_isatools_model(self) -> DataFile:
        return DataFile(
            filename=self.name,
            label=self.type,
        )
