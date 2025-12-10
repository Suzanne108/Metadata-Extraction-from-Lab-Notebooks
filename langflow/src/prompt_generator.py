from enum import Enum

class Prompt(Enum):
    SAMPLE_WORKFLOW = "Generate a short JSON file adhering to the ISA schema for life sciences experiments. The values of the JSON can be entirely random. Please only output the JSON string so that I can parse it immediately. Do not output any additional text"
    PROTOTYPE_V1 = "something"
    PROTOTYPE_V2 = "something else"

