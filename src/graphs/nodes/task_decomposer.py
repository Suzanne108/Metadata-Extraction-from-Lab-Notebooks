from typing import List

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt
from src.task import Task


def create_task_decomposer(model: ChatOpenAI, system_prompt: str, inference_prompt_template: str):

    def task_decomposer(state: ISAExtractionState) -> dict:
        overview_text = state.get("document_overview", "") or ""
        prompt = render_prompt(
            inference_prompt_template,
            {
                "document_overview": overview_text,
            },
        )

        # Model is expected to be bound with structured output at workflow construction
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response: List[Task] = model.invoke(messages)

        if response.tasks:
            lines: List[str] = [f"Planned {len(response.tasks)} tasks:"]
            for idx, task in enumerate(response.tasks, start=1):
                # Log full details for each task so they appear in the streaming logs
                lines.append(
                    f"  {idx}. kind={task.kind}, target={task.target}, status={task.status}, "
                    f"instruction={task.instruction}"
                )
            message_summary = "\n".join(lines)
        else:
            message_summary = "No tasks parsed."

        print(response)

        return {
            "tasks": response.tasks,
            "messages": [AIMessage(content=message_summary)],
        }

    return task_decomposer
