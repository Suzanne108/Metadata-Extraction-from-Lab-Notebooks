from enum import Enum
from pathlib import Path

class Schema(Enum):

    def __new__(cls, display_name: str, file_name: str):
        obj = object.__new__(cls)
        obj._value_ = display_name  # keep the existing display string as .value
        obj.file_name = file_name   # attach a file_name attribute
        return obj

    ASSAY = ("Assay", "assay_schema.json")
    COMMENT = ("Comment", "comment_schema.json")
    DATA = ("Data", "data_schema.json")
    FACTOR = ("Factor", "factor_schema.json")
    FACTOR_VALUE = ("Factor Value", "factor_value_schema.json")
    INVESTIGATION = ("Investigation", "investigation_schema.json")
    MATERIAL_ATTRIBUTE = ("Material Attribute", "material_attribute_schema.json")
    MATERIAL_ATTRIBUTE_VALUE = ("Material Attribute Value", "material_attribute_value_schema.json")
    MATERIAL = ("Material", "material_schema.json")
    ONTOLOGY_ANNOTATION = ("Ontology Annotation", "ontology_annotation_schema.json")
    ONTOLOGY_SOURCE_REFERENCE = ("Ontology Source Reference", "ontology_source_reference_schema.json")
    ORGANIZATION = ("Organization", "organization_schema.json")
    PERSON = ("Person", "person_schema.json")
    PROTOCOL = ("Protocol", "protocol_schema.json")
    PUBLICATION = ("Publication", "publication_schema.json")
    SAMPLE = ("Sample", "sample_schema.json")
    SOURCE = ("Source", "source_schema.json")
    STUDY = ("Study", "study_schema.json")

    def get_file_name(self) -> str:
        """
        Return the JSON schema file name associated with this enum member.
        """
        return self.file_name

    def get_file_path(self) -> str:
        
        return Path.cwd() / "instructions" / "schemas" / self.get_file_name()
