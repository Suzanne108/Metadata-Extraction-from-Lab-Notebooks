from langflow.custom import Component
from langflow.template import Output
from langflow.io import MessageTextInput
from langflow.schema import Data
import json
from isatools import isajson


class ISASchemaChecker(Component):
    display_name = "Validate ISA Schema ISATools"
    description = "This component verifies if a given JSON file conforms to the ISA schema."
    icon = "list-checks"
    name = "ISASchemaCheckerISATools"
    
    iteration = 0


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
        
        temp_isa_file = 'generated_isa_' + str(self.iteration) + '.json'
        json.dump(isa_json, open(temp_isa_file, 'w'))

        json_report = isajson.validate(open(temp_isa_file))
        isa_report = 'isa_validation_report_' + str(self.iteration) + '.json'
        with open(isa_report, 'w') as f:
            json.dump(json_report, f)
        
        self.iteration = self.iteration + 1

        return Data(data=json_report)


    


    
        

















    


    
        














