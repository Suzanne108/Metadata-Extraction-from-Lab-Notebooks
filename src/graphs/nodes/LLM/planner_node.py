import json

from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graphs.nodes.base_node import LLMGraphNode
from src.graphs.types.GraphComponentId import GraphComponentId
from src.graphs.types.Action import WorkflowAction
from src.graphs.types.NextStep import NextStep, NextStepLLM
from src.graphs.utils import trimmed_messages
from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt

class PlannerNode(LLMGraphNode):
    """LLM node that decides the next workflow action."""

    def __init__(self, model: ChatOpenAI, system_prompt: str, inference_prompt_template: str, policy_prompt: str) -> None:
        super().__init__(
            node_id=GraphComponentId.PLANNER_AGENT.value,
            model=model,
            system_prompt=system_prompt,
            inference_prompt_template=inference_prompt_template,
            policy_prompt=policy_prompt,
        )
    

    def _to_internal(self, r) -> NextStep:
        if isinstance(r, NextStepLLM):
            return NextStep(
                action=WorkflowAction[r.action.name],
                instruction=r.instruction,
                query=r.query,
                k=r.k,
                rationale=r.rationale,
            )
        return r  # already NextStep from guardrails

    def _create_planner_message(self, response) -> AIMessage:
        "Creates an AIMessage from the NextStep response"

        content = f"Next Action: {response.action.value}\n"
        if response.instruction:
            content += f"Instruction: {response.instruction}\n"
        if response.query:
            content += f"Query: {response.query}\n"
        if response.k:
            content += f"K: {response.k}\n"
        content += f"Rationale: {response.rationale}\n"

        return AIMessage(content=content)


    def _all_tasks_completed(self, state: ISAExtractionState) -> bool:
        tasks = state.get("tasks") or []
        if not tasks:
            return False
        try:
            return all(getattr(t, "status", None) == "completed" for t in tasks)
        except Exception:
            return False

    def _format_latest_evidence_for_planner(self, latest_evidence):
        if not latest_evidence:
            return "None."
        if not isinstance(latest_evidence, list):
            latest_evidence = [latest_evidence]
        chunks = []
        for idx, item in enumerate(latest_evidence, start=1):
            text = getattr(item, "page_content", None)
            if text is None and isinstance(item, dict):
                text = item.get("page_content") or item.get("content")
            if text is None:
                text = str(item)
            text = str(text).strip()
            if not text:
                continue
            chunks.append(f"{idx}. {text}")
        if not chunks:
            return "None."
        return "\n".join(chunks)

    def _apply_deterministic_guardrails(self, state: ISAExtractionState) -> dict | None:
        """Return an immediate planner decision when pre-LLM guardrails apply."""
        last_action = state.get("last_action")
        tasks_done = self._all_tasks_completed(state)

        # If all tasks are completed → STOP
        if tasks_done:
            response = NextStep(action=WorkflowAction.STOP, instruction=None, query=None, k=None, rationale="guardrail: all tasks completed → END")
            return {
                "messages": trimmed_messages(state, [self._create_planner_message(response)]),
                "planner_decision": response,
            }

        # After UPDATE_DRAFT → Next task
        if last_action == WorkflowAction.UPDATE_DRAFT:
            response = NextStep(action=WorkflowAction.FINISH_TASK, instruction=None, query=None, k=None, rationale="guardrail: next task after update")
            return {
                "messages": trimmed_messages(state, [self._create_planner_message(response)]),
                "planner_decision": response,
            }

        return None

    def _collect_previous_messages(self, state: ISAExtractionState) -> str:
        """Collect and serialize recent messages for planner context."""
        previous_messages = []

        # Get the previous 3 messages for context
        for msg in state.get("messages", [])[-3:]:
            if isinstance(msg, ToolMessage):
                previous_messages.append(f"Message type: ToolMessage, Content: {msg.content}, tool_id: {msg.tool_call_id}")
            elif isinstance(msg, AIMessage):
                previous_messages.append(f"Message type: AIMessage, Content: {msg.content}")

        previous_messages = previous_messages or ["No previous messages."]
        return json.dumps(previous_messages, indent=2)

    def _determine_current_task(self, state: ISAExtractionState) -> str:
        """Serialize the current task for planner prompt context."""
        tasks = state.get("tasks") or []
        idx = state.get("current_task_index")
        if isinstance(idx, int) and 0 <= idx < len(tasks):
            t = tasks[idx]
            current_task = {"kind": t.kind, "target": t.target, "instruction": t.instruction, "status": t.status}
            return json.dumps(current_task, indent=2)
        return "None"

    def _prepare_latest_evidence(self, state: ISAExtractionState) -> tuple[str, str]:
        """Prepare latest evidence text and query values for planner prompt context."""
        raw_latest_evidence = state.get("latest_evidence") or []
        latest_evidence_str = self._format_latest_evidence_for_planner(raw_latest_evidence)
        latest_evidence_query = state.get("latest_evidence_query") or "None."
        return latest_evidence_str, latest_evidence_query

    def run(self, state: ISAExtractionState ) -> dict:
        "Decides next action based on the current state"

        # Deterministic guardrails before calling the LLM
        guardrail_update = self._apply_deterministic_guardrails(state)
        if guardrail_update is not None:
            return guardrail_update

        previous_messages = self._collect_previous_messages(state)
        current_task_str = self._determine_current_task(state)
        latest_evidence_str, latest_evidence_query = self._prepare_latest_evidence(state)

        # Inject the collected context into the planner prompt template
        prompt_template = self.inference_prompt_template
        prompt = render_prompt(
            prompt_template,
            {
                "previous_messages": previous_messages,
                "current_task": current_task_str,
                "latest_evidence": latest_evidence_str,
                "latest_evidence_query": latest_evidence_query,
            },
        )

        messages = [SystemMessage(content=self.system_prompt), SystemMessage(content=self.policy_prompt), HumanMessage(content=prompt)]

        # Call LLM to analyze state and decide next step
        response_llm = self.model.invoke(messages)
        
        # Transform the LLM response into the internal NextStep representation
        response_internal = self._to_internal(response_llm)

        # Update state with the planner decision and the corresponding message
        return {
            "messages": [self._create_planner_message(response_internal)],
            "planner_decision": response_internal
        }
