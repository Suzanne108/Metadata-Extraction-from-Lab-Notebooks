from enum import Enum
from pathlib import Path
import time
import os
from src.schema import Schema
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

from src.document_processing import DocumentManager
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from IPython.display import display, HTML, Image
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from typing import List
import json


# ---- Logging helpers ----

def _normalize_task_obj(t) -> dict:
    if hasattr(t, "dict"):
        return t.dict()
    if isinstance(t, dict):
        return t
    return {"raw": str(t)}


def _task_signature(tdict: dict) -> str:
    kind = tdict.get("kind", "?")
    target = tdict.get("target", "?")
    instr = tdict.get("instruction", "")
    return f"{kind}::{target}::{instr[:80]}"


def _diff_tasks(prev_keys: set[str], curr_list: List) -> tuple[str, set[str]]:
    try:
        curr_keys = {_task_signature(_normalize_task_obj(t)) for t in (curr_list or [])}
        added = curr_keys - prev_keys
        removed = prev_keys - curr_keys
        lines = []
        if added:
            lines.append("   Tasks added:")
            for k in list(added)[:10]:
                lines.append(f"      + {k}")
            if len(added) > 10:
                lines.append(f"      (+{len(added)-10} more)")
        if removed:
            lines.append("   Tasks removed:")
            for k in list(removed)[:10]:
                lines.append(f"      - {k}")
            if len(removed) > 10:
                lines.append(f"      (+{len(removed)-10} more)")
        return ("\n".join(lines) + ("\n" if lines else ""), curr_keys)
    except Exception:
        return ("", prev_keys)


def _truncate(val, limit: int = 400) -> str:
    s = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
    return s if len(s) <= limit else s[:limit] + "..."


def _deep_diff_dicts(old: dict, new: dict, max_depth: int = 3, max_items: int = 100) -> List[str]:
    lines = []

    def walk(o, n, path: List[str], depth: int, budget: List[int]):
        if budget[0] <= 0 or depth > max_depth:
            return
        o_is_dict = isinstance(o, dict)
        n_is_dict = isinstance(n, dict)
        if o_is_dict and n_is_dict:
            o_keys = set(o.keys())
            n_keys = set(n.keys())
            for k in sorted(n_keys - o_keys):
                if budget[0] <= 0:
                    break
                p = ".".join(path + [str(k)])
                lines.append(f"      + {p} = {_truncate(n[k])}")
                budget[0] -= 1
            for k in sorted(o_keys - n_keys):
                if budget[0] <= 0:
                    break
                p = ".".join(path + [str(k)])
                lines.append(f"      - {p} (removed)")
                budget[0] -= 1
            for k in sorted(o_keys & n_keys):
                if budget[0] <= 0:
                    break
                walk(o[k], n[k], path + [str(k)], depth + 1, budget)
            return
        # For lists, compare element-wise up to some length
        if isinstance(o, list) and isinstance(n, list):
            max_len = min(max_items, max(len(o), len(n)))
            for i in range(max_len):
                if budget[0] <= 0:
                    break
                ov = o[i] if i < len(o) else "<missing>"
                nv = n[i] if i < len(n) else "<missing>"
                if json.dumps(ov, sort_keys=True, ensure_ascii=False) != json.dumps(nv, sort_keys=True, ensure_ascii=False):
                    p = ".".join(path + [f"[{i}]"])
                    lines.append(f"      * {p}: {_truncate(ov)} -> {_truncate(nv)}")
                    budget[0] -= 1
            if len(n) > max_len:
                p = ".".join(path + ["[...]"])
                lines.append(f"      + {p} (truncated, {len(n)-max_len} more items)")
            if len(o) > max_len:
                p = ".".join(path + ["[...]"])
                lines.append(f"      - {p} (truncated, {len(o)-max_len} more items)")
            return
        # Fallback: simple change
        if json.dumps(old, sort_keys=True, ensure_ascii=False) != json.dumps(new, sort_keys=True, ensure_ascii=False):
            p = ".".join(path)
            lines.append(f"      * {p}: {_truncate(old)} -> {_truncate(new)}")
            budget[0] -= 1

    walk(old or {}, new or {}, [], 0, [max_items])
    return lines


def invoke_model_with_tools(model, messages: list, tools: List, max_iterations: int = 10):
    """
    Invokes a model with tool-calling capability, handling the tool execution loop.
    
    Args:
        model: The language model to invoke (should be bound with tools)
        messages: Initial list of messages to send to the model
        tools: List of tool objects that can be invoked
        max_iterations: Maximum number of tool-calling iterations to prevent infinite loops
    
    Returns:
        The final response from the model after all tool calls are resolved
    """
    # Create a map of tool names to tool functions for easy lookup
    tool_map = {tool.name: tool for tool in tools}
    
    iteration = 0
    response = None
    
    while iteration < max_iterations:
        iteration += 1
        response = model.invoke(messages)
        
        # Check if the response contains tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Add the AI response with tool calls to messages
            messages.append(response)
            
            # Execute each tool call and add results
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})
                tool_call_id = tool_call['id']
                
                # Look up the tool function
                if tool_name in tool_map:
                    try:
                        # Invoke the tool
                        tool_result = tool_map[tool_name].invoke(tool_args)
                        
                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id
                        ))
                    except Exception as e:
                        print(f"Error invoking tool {tool_name}: {e}")
                        # Add error message as tool response
                        messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call_id
                        ))
                else:
                    print(f"Warning: Tool {tool_name} not found in tool_map. Available: {list(tool_map.keys())}")
                    # Still need to add a response for this tool call
                    messages.append(ToolMessage(
                        content=f"Error: Tool {tool_name} not found",
                        tool_call_id=tool_call_id
                    ))
        else:
            # No more tool calls, we have the final response
            break
    
    # If we hit max iterations without a final response
    if iteration >= max_iterations:
        print(f"Warning: Hit max iterations ({max_iterations}) in tool-calling loop")
    
    return response


class Action(Enum):
    SEARCH_DOCUMENT = "search contextual document"
    REQUEST_SCHEMA = "schema agent"
    VALIDATE = "validate current version"
    UPDATE_DRAFT = "ISA drafting agent"
    FINISH_TASK = "task progress"
    STOP = "END"

class Node(Enum):
    ISA_OVERVIEW = "ISA overview"
    DOC_OVERVIEW = "document overview"
    TASK_DECOMPOSER = "task planning"
    SEARCH_DOCUMENT = "search contextual document"
    SCHEMA_SUMMARIZER_AGENT= "schema agent"
    TASK_PROGRESS = "task progress"
    VALIDATE = "validate current version"
    UPDATE_DRAFT_AGENT = "ISA drafting agent"
    PLANNER_AGENT = "planner agent"
    STOP = "END"


class NextStep(BaseModel):
    action: Action = Field(description= "The next step to perform")
    instruction: str | None = Field(description= "In case of action update_draft, the instruction to be given to the ISA updater agent")
    query: str | None = Field(description= "In case of action search_document, the query to search the document with")
    k: int | None = Field(description= "In case of search_document, the amount of semantically relevant pieces to retrieve")
    rationale: str = Field(description= "The reason for choosing this action")


def index_document() -> DocumentManager:
    path = Path.cwd() / "papers" / "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf "
    doc_manager = DocumentManager()

    doc_manager.embed_documents(path)

    return doc_manager

def stream_messages(agent, state, config: RunnableConfig = None, output_path: str | Path = None, buffer_size: int = 10, flush_interval: float = 5.0):
    """
    Stream agent steps to stdout and also append them efficiently to `output_path`.

    - `output_path`: file to append log lines to (default: ./output/stream_log.txt)
    - `buffer_size`: number of lines to buffer before writing to disk
    - `flush_interval`: maximum seconds between flushes (even if buffer not full)
    """
    print("Streaming agent steps:\n")

    if output_path is None:
        output_path = Path.cwd() / "output" / "stream_log.txt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = []
    last_flush = time.time()

    f = output_path.open("a", encoding="utf-8", buffering=1)  # line-buffered; we'll control flush explicitly
    try:
        # Track state to capture deltas for tasks and draft changes across updates
        last_task_keys: set[str] = set()
        last_draft_repr: str | None = None
        last_draft_obj: dict | None = None
        last_draft_top_hash: dict[str, str] = {}

        for chunk in agent.stream(state, stream_mode="updates", config=config):
            for node_name, data in chunk.items():
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                header = f"[{timestamp}] Step: {node_name}\n"
                print(f"Step: {node_name}")

                # Compose the message content to both print and log
                body = ""
                if "messages" in data:
                    message = data["messages"][-1]
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        tc = message.tool_calls[0]
                        tool_line = f"   Tool call: {tc['name']}\n"
                        args_line = f"   Arguments: {tc.get('args')}\n"
                        body += tool_line + args_line
                        print(tool_line.strip())
                        print(args_line.strip())
                        print()
                    elif hasattr(message, "content"):
                        content = message.content
                        #preview = content[:2000] + "..." if len(content) > 2000 else content
                        content_line = f"   Content: {content}\n"
                        body += content_line
                        print(content_line.strip())
                        print()
                else:
                    body += "   (no messages)\n"

                # Detect and log newly added tasks (if tasks list is present in the update)
                if "tasks" in data:
                    lines, last_task_keys = _diff_tasks(last_task_keys, data.get("tasks"))
                    if lines:
                        body += lines

                # Detect and log changes to draft_isa_metadata (top-level diff summary)
                if "draft_isa_metadata" in data:
                    try:
                        new_draft = data.get("draft_isa_metadata") or {}
                        new_repr = json.dumps(new_draft, sort_keys=True, ensure_ascii=False)
                        if new_repr != last_draft_repr:
                            # Top-level summary
                            new_keys = set(new_draft.keys()) if isinstance(new_draft, dict) else set()
                            old_keys = set(last_draft_top_hash.keys()) if isinstance(last_draft_top_hash, dict) else set()
                            added_keys = new_keys - old_keys
                            removed_keys = old_keys - new_keys
                            changed_keys = []
                            shared = new_keys & old_keys
                            for k in list(shared)[:100]:
                                v_repr = json.dumps(new_draft.get(k), sort_keys=True, ensure_ascii=False)
                                if last_draft_top_hash.get(k) != v_repr:
                                    changed_keys.append(k)

                            body += "   Draft ISA metadata updated:\n"
                            if added_keys:
                                body += f"      + added keys: {sorted(list(added_keys))[:10]}\n"
                            if removed_keys:
                                body += f"      - removed keys: {sorted(list(removed_keys))[:10]}\n"
                            if changed_keys:
                                body += f"      * changed keys: {sorted(changed_keys)[:10]}\n"

                            # Deeper diff (limited depth and items)
                            deep_lines = _deep_diff_dicts(
                                last_draft_obj if isinstance(last_draft_obj, dict) else {},
                                new_draft if isinstance(new_draft, dict) else {},
                                max_depth=3,
                                max_items=80,
                            )
                            if deep_lines:
                                body += "      Deep diff (truncated):\n" + "\n".join(deep_lines[:80]) + "\n"

                            # Update trackers
                            last_draft_repr = new_repr
                            last_draft_obj = new_draft if isinstance(new_draft, dict) else None
                            last_draft_top_hash = {}
                            if isinstance(new_draft, dict):
                                for k, v in list(new_draft.items())[:500]:
                                    try:
                                        last_draft_top_hash[k] = json.dumps(v, sort_keys=True, ensure_ascii=False)
                                    except Exception:
                                        last_draft_top_hash[k] = str(v)
                    except Exception:
                        pass

                # Add a blank line separator for readability
                footer = "\n"

                # Buffer the lines
                buffer.append(header)
                buffer.append(body)
                buffer.append(footer)

                # Decide whether to flush the buffer
                now = time.time()
                if len(buffer) >= buffer_size or (now - last_flush) >= flush_interval:
                    f.write("".join(buffer))
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        # fsync can fail on some filesystems; we ignore to avoid crashing the stream
                        pass
                    buffer.clear()
                    last_flush = now

        # End for loop; flush any remaining buffered lines
        if buffer:
            f.write("".join(buffer))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
            buffer.clear()

    except Exception:
        # On error, ensure buffered content is written before propagating
        if buffer:
            try:
                f.write("".join(buffer))
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        raise
    finally:
        try:
            f.close()
        except Exception:
            pass


def visualise_graph(app: StateGraph, output_path: str | Path = None):
    """Visualise the state graph and save to output_path"""
    
    if output_path is None:
        output_path = Path.cwd() / "output" / "graph_visualization.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate the graph image
    graph_image = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    
    # Save to file
    with open(output_path, "wb") as f:
        f.write(graph_image)
    
    print(f"Graph visualization saved to: {output_path}")
    