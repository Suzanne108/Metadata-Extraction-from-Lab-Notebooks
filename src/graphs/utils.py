from pathlib import Path
import time
import os

from src.document_processing import DocumentManager


def index_document() -> DocumentManager:
    path = Path.cwd() / "papers" / "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf "
    doc_manager = DocumentManager()

    doc_manager.embed_documents(path)

    return doc_manager

def stream(agent, state, output_path: str | Path = None, buffer_size: int = 10, flush_interval: float = 5.0):
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
        for chunk in agent.stream(state, stream_mode="updates"):
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
                        preview = content[:1000] + "..." if len(content) > 1000 else content
                        content_line = f"   Content: {preview}\n"
                        body += content_line
                        print(content_line.strip())
                        print()
                else:
                    body += "   (no messages)\n"

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