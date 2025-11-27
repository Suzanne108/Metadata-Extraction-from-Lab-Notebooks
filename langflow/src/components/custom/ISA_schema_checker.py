from langflow.custom import Component
from langflow.template import Output
from langflow.io import MessageTextInput
from langflow.schema import Data
import json
from isatools import isajson


class ISASchemaChecker(Component):
    display_name = "Validate ISA Schema"
    description = "This component verifies if a given JSON file conforms to the ISA schema."
    icon = "list-checks"
    name = "ISASchemaChecker"


    inputs = [
        MessageTextInput(
            display_name="Agent generated ISA JSON",
            name="isa_json_generated",
            tool_mode=True,
            required=True
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
        
        temp_isa_file = 'temp_isa.json'
        json.dump(isa_json, open(temp_isa_file, 'w'))

        json_report = isajson.validate(open(temp_isa_file))
        with open('isa_validation_report.json', 'w') as f:
            json.dump(json_report, f)

        return Data(data=json_report)


    


    
        














