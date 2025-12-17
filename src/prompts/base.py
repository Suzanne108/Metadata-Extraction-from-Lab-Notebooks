from dataclasses import dataclass
from typing import List


@dataclass
class Prompt:
    """Definition of a prompt."""

    name: str
    version: str
    template: str
    input_variables: List[str]
    #TODO: we could add fields here to define the output schema.