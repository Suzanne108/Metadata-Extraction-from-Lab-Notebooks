from langchain.tools import tool
from src.document_processing import DocumentManager
from langchain_core.documents import Document
from typing import List, Dict
from src.schema import Schema
import json
from jsonschema import validate, ValidationError

def create_doc_getter(doc_manager: DocumentManager) -> List:
    """Creates tools with access to their dependencies"""

    @tool
    def get_doc_chunk(query: str, k: int = 1) -> List[Document]:
        """ Retrieves relevant chunks from the extraction document """

        # The doc manager must be retrieved from the context
        chunks = doc_manager.retrieve_semantic_chunk(query, k)

        return chunks

    return get_doc_chunk

def create_schema_getter() -> List:
    """Creates individual tools for each schema"""
    tools = []
    
    for schema in Schema:

        @tool(f"get_{schema.name.lower()}_schema")
        def get_schema(s: Schema = schema) -> Dict:  # Capture schema in closure
            """Fetches the {schema.value} schema"""

            try:
                file_path = s.get_file_path()
                with open(file_path, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception as e:
                raise ValueError(f"Failed to load schema JSON: {e}")
        
        tools.append(get_schema)
    
    return tools

@tool
def validate_ISA_schema(isa: dict) -> Dict:
    """ Validates the given ISA schema, returns a report """

    with open(Schema.INVESTIGATION.get_file_path(), "r", encoding="utf-8") as fp:
        schema = json.load(fp)
    
    try: 
            
        validate(instance=isa, schema=schema)
        report = {"valid": True, "message": "JSON is valid against the schema."}

        return report 
        
    except ValidationError as e:

        report = {"valid": False, "message": e.message, "json_path": e.json_path, "context": e.context, "cause": e.cause}

        return report

    