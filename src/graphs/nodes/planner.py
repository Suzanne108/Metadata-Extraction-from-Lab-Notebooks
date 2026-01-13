from src.graphs.utils import NextStep
from src.state.v3 import ISAExtractionStateV3
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.prompts.renderer import render_prompt
from src.prompts.registry import PromptRegistry


import json

#TODO: refactor so that the inference prompt can be given as argument

def create_planner(model: ChatOpenAI, system_prompt: str, policy_prompt: str):

    def create_planner_message(response: NextStep, prompt: str) -> AIMessage:
        "Creates an AIMessage from the NextStep response"

        received_prompt = f"Received information:\n{prompt}\n\n"
        
        content = f"Next Action: {response.action.value}\n"
        if response.instruction:
            content += f"Instruction: {response.instruction}\n"
        if response.query:
            content += f"Query: {response.query}\n"
        if response.k:
            content += f"K: {response.k}\n"
        content += f"Rationale: {response.rationale}\n"

        return AIMessage(content=content + "\n" + received_prompt)


    def planner(state: ISAExtractionStateV3 ) -> dict:
        "Decides next action based on the current state"

        vr = state.get("validation_result") or {}
        vr = json.dumps(vr, indent=2)
        draft = state.get("draft_isa_metadata") or {}
        draft = json.dumps(draft, indent=2)
        evidence = list(state.get("evidence").keys()) or []
        evidence = json.dumps(evidence, indent=2)

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
        prompt_template = registry.get_prompt("planner", "v3-inference")
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

        # Is of type NextStep
        response = model.invoke(messages)

        return {
            "messages": state.get("messages", []) + [create_planner_message(response, prompt)],
            "orchestrator_decision": response
        }

    return planner