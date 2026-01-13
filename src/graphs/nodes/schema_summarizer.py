

from src.state.v3 import ISAExtractionStateV3
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from src.prompts.renderer import render_prompt
from src.graphs.utils import invoke_model_with_tools

def create_schema_summarizer(model: ChatOpenAI, schema_getter_tools: list, system_prompt: str, inference_prompt_template: str):


    def summarize_schema(state: ISAExtractionStateV3) -> dict:
        "Fetches and summarizes the required schema"

        instruction = state['orchestrator_decision'].instruction

        prompt = render_prompt(
            inference_prompt_template,
            {
                "instruction": instruction
            }
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        # Invoke model with tool-calling support
        response = invoke_model_with_tools(model, messages, schema_getter_tools, max_iterations=10)
        
        print("Schema summarizer response:", response.content)

        return {
            "messages": state.get("messages", []) +  [response]
        }
        
    return summarize_schema