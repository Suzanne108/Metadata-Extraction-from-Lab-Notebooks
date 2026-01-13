from typing import List, Optional
from langchain.messages import AIMessage
from src.state.v3 import ISAExtractionStateV3, Task


def create_task_progressor():
    def task_progressor(state: ISAExtractionStateV3) -> dict:
        tasks: Optional[List[Task]] = state.get("tasks") or []
        messages = list(state.get("messages", []))
        if not tasks:
            messages.append(AIMessage(content="NODE: TASK_PROGRESS; status: no-tasks; message: No tasks available to progress."))
            return {"messages": messages}

        idx = state.get("current_task_index")
        if idx is None:
            idx = 0

        # Clamp to bounds and find first non-completed if out of range
        if idx < 0 or idx >= len(tasks):
            idx = next((i for i, t in enumerate(tasks) if t.status != "completed"), 0)

        decision = state.get("orchestrator_decision")
        last_action = getattr(decision, "action", None)
        last_action_val = getattr(last_action, "value", None)

        def set_status(i: int, status: str):
            t = tasks[i]
            tasks[i] = Task(kind=t.kind, target=t.target, instruction=t.instruction, status=status)

        # Initial entry after decomposition: ensure first task is in-progress
        if last_action_val is None:
            if tasks[idx].status == "pending":
                set_status(idx, "in-progress")
                messages.append(AIMessage(content=f"NODE: TASK_PROGRESS; status: in-progress; task_index: {idx}; total: {len(tasks)}; kind: {tasks[idx].kind}; target: {tasks[idx].target}"))
            else:
                messages.append(AIMessage(content=f"NODE: TASK_PROGRESS; status: continue; task_index: {idx}; total: {len(tasks)}; kind: {tasks[idx].kind}; target: {tasks[idx].target}"))
            return {"current_task_index": idx, "tasks": tasks, "messages": messages}

        # Set current task completed and next task in progress

        if last_action_val == "task progress":

            if 0 <= idx < len(tasks) and tasks[idx].status != "completed":
                set_status(idx, "completed")
                messages.append(AIMessage(content=f"NODE: TASK_PROGRESS; status: completed; task_index: {idx}; total: {len(tasks)}; kind: {tasks[idx].kind}; target: {tasks[idx].target}"))
                # Advance to next pending task
                next_idx = next((i for i in range(idx + 1, len(tasks)) if tasks[i].status != "completed"), None)
                if next_idx is not None:
                    if tasks[next_idx].status == "pending":
                        set_status(next_idx, "in-progress")
                    messages.append(AIMessage(content=f"NODE: TASK_PROGRESS; status: in-progress; task_index: {next_idx}; total: {len(tasks)}; kind: {tasks[next_idx].kind}; target: {tasks[next_idx].target}"))
                return {"current_task_index": next_idx, "tasks": tasks, "messages": messages}

            messages.append(AIMessage(content="NODE: TASK_PROGRESS; status: noop; message: Finish requested, but no active task to complete."))
            return {"current_task_index": idx, "tasks": tasks, "messages": messages}

        # Default: no state transition, echo current
        messages.append(AIMessage(content=f"NODE: TASK_PROGRESS; status: working; task_index: {idx}; total: {len(tasks)}; kind: {tasks[idx].kind}; target: {tasks[idx].target}"))
        return {"current_task_index": idx, "tasks": tasks, "messages": messages}

    return task_progressor
