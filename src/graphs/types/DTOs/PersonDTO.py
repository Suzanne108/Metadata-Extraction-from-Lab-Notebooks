from typing import List, Optional

from pydantic import BaseModel, Field

from isatools.model import Person


class PersonDTO(BaseModel):
    uid: Optional[str] = Field(default=None, description="Stable identifier alias; falls back to id if provided")
    first_name: str = Field(description="Given name")
    last_name: str = Field(description="Family name")
    email: Optional[str] = Field(None, description="Contact email")
    mid_initials: Optional[str] = Field(None, description="Middle initials")
    address: Optional[str] = Field(None, description="address")
    affiliation: Optional[str] = Field(None, description="Organization or lab")

    def to_isatools_model(self) -> Person:
        return Person(
            first_name=self.first_name,
            last_name=self.last_name,
            email=self.email,
            affiliation=self.affiliation,
            address=self.address,
            mid_initials=self.mid_initials,
        )


