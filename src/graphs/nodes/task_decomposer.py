from src.state.v3 import ISAExtractionStateV3, TaskList
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts.renderer import render_prompt


def create_task_decomposer(model: ChatOpenAI, system_prompt: str, inference_prompt_template: str):

    def task_decomposer(state: ISAExtractionStateV3) -> dict:
        overview_text = state.get("document_overview", "") or ""
        prompt = render_prompt(
            inference_prompt_template,
            {
                "document_overview": overview_text,
            },
        )

        # Model is expected to be bound with structured output at workflow construction
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response : TaskList = model.invoke(messages)

        preview = response.tasks[:3]
        message_summary = f"Planned {len(response.tasks)} tasks.\nPreview: {preview}" if response.tasks else "No tasks parsed."

        print(response)

        return {
            "tasks": response.tasks,
            "messages": [AIMessage(content=message_summary)],
        }

    return task_decomposer
