from langflow.custom import Component
from langflow.template import Output
from langflow.io import DropdownInput, FileInput
from langflow.schema import Data
import json
from enum import Enum

class Schema(Enum):
    ASSAY = "Assay"
    COMMENT = "Comment"
    DATA = "Data"
    FACTOR = "Factor"
    FACTOR_VALUE = "Factor Value"
    INVESTIGATION = "Investigation"
    MATERIAL_ATTRIBUTE = "Material Attribute"
    MATERIAL_ATTRIBUTE_VALUE = "Material Attribute Value"
    MATERIAL = "Material"
    ONTOLOGY_ANNOTATION = "Ontology Annotation"
    ONTOLOGY_SOURCE_REFERENCE = "Ontology Source Reference"
    ORGANIZATION = "Organization"
    PERSON = "Person"
    PROTOCOL = "Protocol"
    PUBLICATION = "Publication"
    SAMPLE = "Sample"
    SOURCE = "Source"
    STUDY = "Study"

class SchemaHelper():

    @staticmethod
    def get_all_schemas() -> list[str]:
        return [schema.value for schema in Schema]

    @staticmethod
    def get_file_name(schema: Schema) -> str:
        return f"{schema.value.lower().replace(' ', '_')}_schema"

    @staticmethod
    def get_schema_by_name(name: str) -> Schema:
        for schema in Schema:
            if schema.value == name:
                return schema
        raise ValueError(f"Schema with name '{name}' not found.")


class GetSchemaTool(Component):
    display_name = "Get Schema Tool"
    description = "This a tool to get an ISA schema by name"
    icon = "folder-search"
    name = "GetSchemaTool"

    helper = SchemaHelper()


    inputs = [
        DropdownInput(
            display_name="Schema Name",
            name="schema_asked",
            options= helper.get_all_schemas(),
            tool_mode=True,
            required=True
        ),
        FileInput(
            display_name="Schema Files",
            name="schema_files",
            required=True,
            is_list=True,
            file_types=["json"],
            
        )
    ]
    outputs = [
        Output(
            display_name="Schema contents",
            name="schema_contents",
            method="get_schema"
        )
    ]
    
    def get_schema(self) -> Data:
        """Load and return the selected schema as JSON."""
        selected_file_name = self.schema_asked
        selected : Schema = SchemaHelper.get_schema_by_name(selected_file_name)
        helper = SchemaHelper()
        file_name = helper.get_file_name(selected)


        match = None

        for f in self.schema_files:
            if file_name in f:
                match = f
                break

        if not match:
            raise ValueError(f"Schema file '{file_name}' not found among uploaded files. Example expected file {self.schema_files[0]}")

        try:
            with open(match, "r", encoding="utf-8") as fp:
                schema = json.load(fp)
            return Data(data=schema)
        except Exception as e:
            raise ValueError(f"Failed to load schema JSON: {e}")








