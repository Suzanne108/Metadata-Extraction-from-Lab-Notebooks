
from pathlib import Path
import time
import os
from typing import List, Any, Dict
import json

from langgraph.graph import StateGraph

from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage

from src.document_processing import DocumentManager
from pydantic import BaseModel
import hashlib


# ---- DTO registry & UID helpers ----

def _model_to_dict_safe(model: BaseModel) -> dict:
    try:
        # Pydantic v2
        return model.model_dump()
    except Exception:
        try:
            return model.dict()
        except Exception:
            return {}


def _natural_key(dto: BaseModel) -> str:
    """Compute a deterministic natural key string based on known fields per DTO type.
    Falls back to a hash of the entire model dict if specifics are unavailable.
    """
    t = type(dto).__name__
    d = _model_to_dict_safe(dto)

    def pick(*keys):
        vals = [str(d.get(k) or "") for k in keys]
        return "::".join(vals)

    if t == "InvestigationDTO":
        key = pick("identifier", "title")
    elif t == "StudyDTO":
        key = pick("identifier", "title")
    elif t == "AssayDTO":
        mt = d.get("measurement_type") or {}
        tt = d.get("technology_type") or {}
        key = "::".join([
            str(d.get("technology_platform") or ""),
            str(mt.get("term") if isinstance(mt, dict) else ""),
            str(tt.get("term") if isinstance(tt, dict) else ""),
        ])
    elif t == "DataDTO":
        key = pick("name", "type")
    elif t == "ProcessDTO":
        key = pick("name", "date", "performer")
    elif t == "ProtocolDTO":
        key = pick("name", "version", "uri")
    elif t == "FactorDTO":
        key = pick("factor_name")
    elif t == "FactorValueDTO":
        key = pick("category_name", "value_text", "unit_text")
    elif t == "SourceDTO":
        key = pick("name")
    elif t == "SampleDTO":
        key = pick("name")
    elif t == "MaterialDTO":
        key = pick("name", "type")
    elif t == "PublicationDTO":
        key = pick("doi", "pubmed_id", "title")
    elif t == "PersonDTO":
        # Prefer explicit id if present
        key = d.get("id") or "::".join([str(d.get("first_name") or ""), str(d.get("last_name") or ""), str(d.get("email") or "")])
    else:
        # Fallback: stable hash of dict
        key = json.dumps(d, sort_keys=True)

    if not key or key.replace(":", "").strip() == "":
        key = json.dumps(d, sort_keys=True)

    # Return a compact uid prefix + hash(key)
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{t}:{h}"


def ensure_uid(dto: BaseModel) -> str:
    """Ensure dto.uid exists;  or compute from natural key."""
    # If uid attribute exists and is set, return
    if hasattr(dto, "uid"):
        val = getattr(dto, "uid", None)
        if val:
            return val
    uid = _natural_key(dto)
    try:
        setattr(dto, "uid", uid)
    except Exception:
        # Ignore if setting fails; still return computed uid
        pass
    return uid


_DTO_REGISTRY_KEY_BY_CLASS: Dict[str, str] = {
    "InvestigationDTO": "registry_investigations",
    "StudyDTO": "registry_studies",
    "AssayDTO": "registry_assays",
    "DataFileDTO": "registry_data",
    "ProcessDTO": "registry_processes",
    "ProtocolDTO": "registry_protocols",
    "FactorDTO": "registry_factors",
    "FactorValueDTO": "registry_factor_values",
    "SourceDTO": "registry_sources",
    "SampleDTO": "registry_samples",
    "MaterialDTO": "registry_materials"
}


def _get_registry_for_dto(state: dict, dto: BaseModel) -> tuple[str, Dict[str, BaseModel]]:
    key = _DTO_REGISTRY_KEY_BY_CLASS.get(type(dto).__name__)
    if not key:
        return "", {}
    reg = dict(state.get(key) or {})
    return key, reg


def register_dto(state: dict, dto: BaseModel) -> dict:
    """Upsert a single DTO into its registry with merge by uid."""
    uid = ensure_uid(dto)
    reg_key, reg = _get_registry_for_dto(state, dto)
    if not reg_key:
        return {}
    # Overwrite policy: replace any existing entry with the new DTO
    reg[uid] = dto
    return {reg_key: reg}


def ensure_dto_uid_and_register(state: dict, dto: BaseModel, max_depth: int = 2) -> dict:
    """Ensure uid and register dto and shallow-nested DTOs (lists or single) into appropriate registries.
    Returns a dict of state updates for all touched registries.
    """
    updates: Dict[str, Any] = {}

    def _accumulate(update: dict):
        for k, v in (update or {}).items():
            if isinstance(v, dict):
                merged = dict(updates.get(k) or {})
                merged.update(v)
                updates[k] = merged
            else:
                updates[k] = v

    # Register the primary dto
    _accumulate(register_dto(state, dto))

    if max_depth <= 0:
        return updates

    # Traverse shallow attributes for nested DTOs
    d = _model_to_dict_safe(dto)
    for attr_name, val in d.items():
        # Check if original attribute is a BaseModel or list of BaseModel via getattr
        try:
            actual = getattr(dto, attr_name, None)
        except Exception:
            actual = None

        if isinstance(actual, BaseModel):
            _accumulate(register_dto(state | updates, actual))
        elif isinstance(actual, list):
            for item in actual:
                if isinstance(item, BaseModel):
                    _accumulate(register_dto(state | updates, item))

    return updates



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

def index_document(document_path) -> DocumentManager:
    doc_manager = DocumentManager()

    doc_manager.embed_documents(document_path)

    return doc_manager

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

                print(f"Invoking tool: {tool_name} with args: {tool_args}")
                
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


def trimmed_messages(state: dict, additions: list | None = None, max_messages: int = 30, max_total_chars: int = 60000):
    """Return a bounded list of messages by appending `additions` then trimming.

    - Keeps only the latest `max_messages` messages
    - Also enforces an approximate character budget across message.contents
    This reduces unbounded growth of the `messages` channel while preserving
    most recent context for routing/diagnostics.
    """
    msgs = list(state.get("messages", []) or [])
    if additions:
        if isinstance(additions, list):
            msgs.extend(additions)
        else:
            msgs.append(additions)

    # Limit by count first
    if len(msgs) > max_messages:
        msgs = msgs[-max_messages:]

    # Then enforce a coarse character budget (drop oldest first)
    def _content_len(m):
        try:
            return len(getattr(m, "content", "") or "")
        except Exception:
            return 0

    total = 0
    kept: list = []
    for m in reversed(msgs):
        total += _content_len(m)
        if total > max_total_chars and kept:
            break
        kept.append(m)
    kept.reverse()
    return kept

def stream_messages(run_id, agent, state, config: RunnableConfig = None, output_path: str | Path = None, buffer_size: int = 10, flush_interval: float = 5.0):
    """
    Stream agent steps to stdout and also append them efficiently to `output_path`.

    - `output_path`: file to append log lines to (default: ./output/stream_log.txt)
    - `buffer_size`: number of lines to buffer before writing to disk
    - `flush_interval`: maximum seconds between flushes (even if buffer not full)
    """
    print("Streaming agent steps:\n")

    if output_path is None:
        output_path = Path.cwd() / "output" / f"stream_log_{run_id}.txt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = []
    last_flush = time.time()

    f = output_path.open("a", encoding="utf-8", buffering=1)  # line-buffered; we'll control flush explicitly
    try:
        # Track state to capture deltas for tasks across updates
        last_task_keys: set[str] = set()
        # Maintain an aggregated view of state to allow full registry snapshots
        aggregated_state: Dict[str, Any] = dict(state or {})

        def _merge_update(base: dict, upd: dict) -> dict:
            # Shallow merge, but for registry_* keys, merge dict values
            out = dict(base)
            for k, v in (upd or {}).items():
                if isinstance(v, dict) and k.startswith("registry_"):
                    merged = dict(out.get(k) or {})
                    merged.update(v)
                    out[k] = merged
                else:
                    out[k] = v
            return out

        def _serialize_model(obj: Any) -> Any:
            try:
                return obj.model_dump()
            except Exception:
                try:
                    return obj.dict()
                except Exception:
                    return str(obj)

        def _registry_snapshot_text(state_like: dict) -> str:
            registry_keys = set(_DTO_REGISTRY_KEY_BY_CLASS.values())
            snapshot: Dict[str, dict] = {}
            for rk in sorted(registry_keys):
                reg = state_like.get(rk) or {}
                if reg:
                    snapshot[rk] = {uid: _serialize_model(obj) for uid, obj in reg.items()}
            if not snapshot:
                return "   Registries snapshot: (empty)\n"
            try:
                return "   Registries snapshot:\n" + json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
            except Exception:
                return "   Registries snapshot: (unserializable)\n"

        for chunk in agent.stream(state, stream_mode="updates", config=config):
            for node_name, data in chunk.items():
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                header = f"[{timestamp}] Step: {node_name}\n"
                print(f"Step: {node_name}")

                # Compose the message content to both print and log
                body = ""
                # Merge current update into aggregated state for later snapshots
                if isinstance(data, dict):
                    aggregated_state = _merge_update(aggregated_state, data)

                if isinstance(data, dict) and "messages" in data:
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
                if isinstance(data, dict)  and "tasks" in data:
                    lines, last_task_keys = _diff_tasks(last_task_keys, data.get("tasks"))
                    if lines:
                        body += lines
                        # On task list change, also snapshot registries
                        body += _registry_snapshot_text(aggregated_state)

                # Also snapshot registries whenever the task progress node runs
                if isinstance(node_name, str) and node_name.strip().lower() == "task progress":
                    body += _registry_snapshot_text(aggregated_state)

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


def visualise_graph(app_or_graph: StateGraph, output_path: str | Path = None):
    """Visualise the graph. Accepts either a compiled app (with get_graph) or an uncompiled StateGraph.

    If given an uncompiled StateGraph, compiles it with an in-memory checkpointer solely for visualization.
    """

    if output_path is None:
        output_path = Path.cwd() / "output" / "graph_visualization.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Obtain a Graph instance with draw_mermaid_png()
    try:
        if hasattr(app_or_graph, "get_graph"):
            graph_obj = app_or_graph.get_graph()
        else:
            # Compile temporarily with an in-memory checkpointer for visualization
            from langgraph.checkpoint.memory import MemorySaver
            compiled = app_or_graph.compile(checkpointer=MemorySaver())
            graph_obj = compiled.get_graph()
    except Exception as e:
        raise RuntimeError(f"Failed to prepare graph for visualization: {e}")

    # Generate the graph image
    graph_image = graph_obj.draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )

    # Save to file
    with open(output_path, "wb") as f:
        f.write(graph_image)

    print(f"Graph visualization saved to: {output_path}")
    

def write_final_state(final_state, run_id: str, output_path: str | Path = None) -> Path:
    if output_path is None:
        output_path = Path.cwd() / "output" / f"isa_draft_{run_id}.json"
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    draft = final_state.get("draft_isa_metadata") if isinstance(final_state, dict) else getattr(final_state, "draft_isa_metadata", {})
    with p.open("w", encoding="utf-8") as f:
        json.dump(draft or {}, f, ensure_ascii=False, indent=2)
    return p
