import requests
from dotenv import load_dotenv
import os
import uuid
import json
from isatools import isajson
from prompt_generator import Prompt


def call_langflow_workflow(url, prompt) -> dict:
    """Calls a Langflow workflow and returns the result as a string."""

    # Request payload configuration
    payload = {
        "output_type": "chat",
        "input_type": "text",
        "input_value": prompt,
        "session_id": str(uuid.uuid4())
    }
    

    load_dotenv()

    # Request headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": os.getenv("LANGFLOW_API_KEY")
    }

    try:
        # Send API request
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")

    return response.json()

def extract_message(data):
    """Extracts the message from the Langflow workflow response data."""

    try:
        return data["outputs"][0]["outputs"][0]["outputs"]["message"]["message"]
    except (KeyError, IndexError):
        return None


def parse_workflow_result(result: str) -> dict :
    """Parses the result from a Langflow workflow and returns it in JSON format.
    This validates that the LLM outputs proper JSON."""

    try:
        return json.loads(result)
    except json.JSONDecodeError:    
        return TypeError("The LLM did not return valid JSON.")
    
def generate_ISA_json(url: str, output_file: str, prompt: Prompt) -> dict:
    """Generates an ISA JSON file using a Langflow workflow."""

    isa_json = parse_workflow_result(extract_message(call_langflow_workflow(url, prompt)))
    with open(output_file, 'w') as f:
       json.dump(isa_json, f)

    return isa_json

def validate_ISA_schema(file_path: str) -> dict :
    """Validates if the given string conforms to the ISA schema."""

    json_report = isajson.validate(open('sample.json'))
    with open('sample_isa_validation_report.json', 'w') as f:
        json.dump(json_report, f)

    return json_report

if __name__=="__main__":
   prompt = Prompt.PROTOTYPE_V2

   url = "http://localhost:7860/api/v1/run/23d139c5-9665-45e5-a612-414b92a9e9c4"

   json_output_file = "prototype_version_2.json"

   generate_ISA_json = (url, json_output_file, prompt)

   validate_ISA_schema(json_output_file)
   

