import json

from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graphs.types.Action import Action
from src.graphs.types.NextStep import NextStep, NextStepLLM
from src.graphs.utils import trimmed_messages
from src.state import ISAExtractionState
from src.prompts.renderer import render_prompt
from src.prompts.registry import PromptRegistry

#TODO: refactor so that the inference prompt can be given as argument
#TODO: why does this need to be wrapped in a create_planner function? Can't I create an object of a Planner class instead? Is there a better way to do this? --> same for other nodes

def create_planner(model: ChatOpenAI, system_prompt: str, policy_prompt: str):

    def to_internal(r) -> NextStep:
        if isinstance(r, NextStepLLM):
            return NextStep(
                action=Action[r.action.name],
                instruction=r.instruction,
                query=r.query,
                k=r.k,
                rationale=r.rationale,
            )
        return r  # already NextStep from guardrails

    def create_planner_message(response) -> AIMessage:
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


    def _all_tasks_completed(state: ISAExtractionState) -> bool:
        tasks = state.get("tasks") or []
        if not tasks:
            return False
        try:
            return all(getattr(t, "status", None) == "completed" for t in tasks)
        except Exception:
            return False
        
    #TODO: rename method to something more descriptive, like "plan_next_step" or "decide_next_action"

    #TODO: break up this method into smaller helper methods for clarity and maintainability. It's currently too long and complex.

    def planner(state: ISAExtractionState ) -> dict:
        "Decides next action based on the current state"

        # Deterministic pre-checks (guardrails) before calling the LLM
        last_action = (state.get("last_action") or "").upper()
        tasks_done = _all_tasks_completed(state)

        # 0) If all tasks are completed → STOP (END)
        if tasks_done:
            response = NextStep(action=Action.STOP, instruction=None, query=None, k=None, rationale="guardrail: all tasks completed → END")
            return {
                "messages": trimmed_messages(state, [create_planner_message(response)]),
                "orchestrator_decision": response
            }

        #After UPDATE_DRAFT → Next task
        if last_action == "UPDATE_DRAFT":
            response = NextStep(action=Action.FINISH_TASK, instruction=None, query=None, k=None, rationale="guardrail: next task after update")
            return {
                "messages": trimmed_messages(state, [create_planner_message(response)]),
                "orchestrator_decision": response
            }

        # Otherwise, call LLM to analyze state and decide next step
        

        previous_messages = []

        # Get the previous 3 messages for context
        for msg in state.get("messages", [])[-3:]:
            if isinstance(msg, ToolMessage):
                previous_messages.append(f"Message type: ToolMessage, Content: {msg.content}, tool_id: {msg.tool_call_id}" )
            elif isinstance(msg, AIMessage):
                previous_messages.append(f"Message type: AIMessage, Content: {msg.content}" )
        
        previous_messages = previous_messages or ["No previous messages."]
        previous_messages = json.dumps(previous_messages, indent=2)

        # Determine current task string for prompt
        tasks = state.get("tasks") or []
        idx = state.get("current_task_index")
        if isinstance(idx, int) and 0 <= idx < len(tasks):
            t = tasks[idx]
            current_task = {"kind": t.kind, "target": t.target, "instruction": t.instruction, "status": t.status}
            current_task_str = json.dumps(current_task, indent=2)
        else:
            current_task_str = "None"

        # Prepare latest evidence (last retrieved document chunks) for the planner
        raw_latest_evidence = state.get("latest_evidence") or []
        # Normalize and stringify similar to specialist agents
        def _format_latest_evidence_for_planner(latest_evidence):
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

        latest_evidence_str = _format_latest_evidence_for_planner(raw_latest_evidence)
        latest_evidence_query = state.get("latest_evidence_query") or "None."

        # Call LLM to analyze state and decide next step
        registry = PromptRegistry.get_instance()
        prompt_template = registry.get_prompt("planner", "v4-inference")
        prompt = render_prompt(
            prompt_template,
            {
                "previous_messages": previous_messages,
                "current_task": current_task_str,
                "latest_evidence": latest_evidence_str,
                "latest_evidence_query": latest_evidence_query,
            },
        )

        messages = [SystemMessage(content=system_prompt), SystemMessage(content=policy_prompt), HumanMessage(content=prompt)]

        # LLM returns NextStepLLM; translate to internal NextStep for state
        response_llm = model.invoke(messages)

        #TODO: why is wrapping in NextStepLLM necessary? Can I just return the LLM response directly? Is there a better way to do this?

        response_internal = to_internal(response_llm)

        # Emit only the new planner message
        return {
            "messages": [create_planner_message(response_internal)],
            "orchestrator_decision": response_internal
        }

    return planner