from typing import List

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from src.graphs.nodes.base_node import LLMGraphNode
from src.graphs.types.Node import GraphComponentId
from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt
from src.task import Task


class TaskDecomposerNode(LLMGraphNode):
    """LLM node that decomposes document overview text into workflow tasks."""

    def __init__(self, model: ChatOpenAI, system_prompt: str, inference_prompt_template: str) -> None:
        super().__init__(
            node_id=GraphComponentId.TASK_DECOMPOSER.value,
            model=model,
            system_prompt=system_prompt,
            inference_prompt_template=inference_prompt_template,
        )

    def run(self, state: ISAExtractionState) -> dict:
        overview_text = state.get("document_overview", "") or ""
        prompt = render_prompt(
            self.inference_prompt_template,
            {
                "document_overview": overview_text,
            },
        )

        # Model is expected to be bound with structured output at workflow construction
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        response: List[Task] = self.model.invoke(messages)

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

