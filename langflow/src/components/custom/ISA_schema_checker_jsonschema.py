from langflow.custom import Component
from langflow.template import Output
from langflow.io import MessageTextInput, FileInput
from langflow.schema import Data
import json
from jsonschema import validate, ValidationError


class ISASchemaChecker(Component):
    display_name = "Validate ISA Schema jsonschema"
    description = "This component verifies if a given JSON file conforms to the ISA schema."
    icon = "list-checks"
    name = "ISASchemaCheckerJSONSchema"
    
    iteration = 0


    inputs = [
        MessageTextInput(
            display_name="Agent generated ISA JSON",
            name="isa_json_generated",
            tool_mode=True,
            required=True
        ),
        FileInput(
            display_name="Schema Files",
            name="schema_files",
            required=True,
            is_list=True,
            file_types=["json"]     
        )

    ]
    outputs = [
        Output(
            display_name="Schema validation report",
            name="schema_validation_report",
            method="validate_schema"
        )
    ]
    
    def validate_schema(self) -> Data:
        """Validates if the given JSON conforms to the ISA schema."""

        try:
            isa_json = json.loads(self.isa_json_generated)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format."}
        
        temp_isa_file = 'generated_isa_' + str(self.iteration) + '.json'

        with open(temp_isa_file, 'w') as f:
            json.dump(isa_json, open(temp_isa_file, 'w'))
        
        file_name = "investigation_schema"
        schema_path = None
        for f in self.schema_files:
        # LangFlow FileInput gives a path as string
            if isinstance(f, str) and file_name in os.path.basename(f):
                schema_path = f
                break

        if schema_path is None:
            return {"error": "Investigation schema file not found."}

        try:
            with open(schema_path, "r", encoding="utf-8") as sf:
                schema = json.load(sf)
        except Exception as e:
            return {"error": f"Failed to load schema: {str(e)}"}

        report_file_name = 'isa_validation_report_' + str(self.iteration) + '.json'

        self.iteration = self.iteration + 1

        try: 
            
            validate(instance=isa_json, schema=schema)
            report = {"valid": True, "message": "JSON is valid against the schema."}

            with open(report_file_name, 'w') as f:
                json.dump(report, f)

            return report 
        
        except ValidationError as e:

            report = {"valid": False, "message": e.message, "json_path": e.json_path, "context": e.context, "cause": e.cause}

            with open(report_file_name, 'w') as f:
                json.dump(report, f)

            return report
            
            
            
    


    


    
        

















    


    
        















    
        

















    


    
        














