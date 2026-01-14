from src.graphs.utils import NextStep, NextStepLLM, Action, trimmed_messages
from src.state.v3 import ISAExtractionStateV3
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.prompts.renderer import render_prompt
from src.prompts.registry import PromptRegistry


import json

#TODO: refactor so that the inference prompt can be given as argument

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

    def create_planner_message(response, draft, current_task_str) -> AIMessage:
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


    def _all_tasks_completed(state: ISAExtractionStateV3) -> bool:
        tasks = state.get("tasks") or []
        if not tasks:
            return False
        try:
            return all(getattr(t, "status", None) == "completed" for t in tasks)
        except Exception:
            return False

    def planner(state: ISAExtractionStateV3 ) -> dict:
        "Decides next action based on the current state"

        # Deterministic pre-checks (guardrails) before calling the LLM
        last_action = (state.get("last_action") or "").upper()
        validation_ok = bool((state.get("validation_result") or {}).get("valid") is True)
        tasks_done = _all_tasks_completed(state)

        vr = state.get("validation_result") or {}
        vr = json.dumps(vr, indent=2)
        draft = state.get("draft_isa_metadata") or {}
        draft = json.dumps(draft, indent=2)
        evidence = list(state.get("evidence").keys()) or []
        evidence = json.dumps(evidence, indent=2)

        # 0) If all tasks are completed → STOP (END)
        if tasks_done:
            response = NextStep(action=Action.STOP, instruction=None, query=None, k=None, rationale="guardrail: all tasks completed → END")
            return {
                "messages": trimmed_messages(state, [create_planner_message(response, draft, "Pre-LLM guardrail")]),
                "orchestrator_decision": response
            }

        # 1) After UPDATE_DRAFT → VALIDATE next
        if last_action == "UPDATE_DRAFT":
            response = NextStep(action=Action.VALIDATE, instruction=None, query=None, k=None, rationale="guardrail: validate after update")
            return {
                "messages": trimmed_messages(state, [create_planner_message(response, draft, "Pre-LLM guardrail")]),
                "orchestrator_decision": response
            }

        # 2) After VALIDATE + valid → FINISH_TASK
        if last_action == "VALIDATE" and validation_ok:
            response = NextStep(action=Action.FINISH_TASK, instruction=None, query=None, k=None, rationale="guardrail: finish after successful validation")
            return {
                "messages": trimmed_messages(state, [create_planner_message(response, draft, "Pre-LLM guardrail")]),
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

        # Call LLM to analyze state and decide next step
        registry = PromptRegistry()
        prompt_template = registry.get_prompt("planner", "v4-inference")
        prompt = render_prompt(
            prompt_template,
            {
                "draft": draft,
                "validation_result": vr,
                "evidence": evidence,
                "previous_messages": previous_messages,
                "current_task": current_task_str,
            },
        )

        messages = [SystemMessage(content=system_prompt), SystemMessage(content=policy_prompt), HumanMessage(content=prompt)]

        # LLM returns NextStepLLM; translate to internal NextStep for state
        response_llm = model.invoke(messages)

        response_internal = to_internal(response_llm)

        # Emit only the new planner message; LangGraph's additive channel will accumulate.
        return {
            "messages": [create_planner_message(response_internal, draft, current_task_str)],
            "orchestrator_decision": response_internal
        }

    return planner