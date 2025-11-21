from providers.base_provider import LLMProvider
from typing import List
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, END, StateGraph
from langsmith import Client
from dotenv import load_dotenv
import os
import json
import textwrap
import shutil
import re

# State controls what data is input to the application, transferred between components, and output from the application.
class OverallState(TypedDict):
    question: str
    context: List[Document]
    answer: str

class InputState(TypedDict):
    question: str
    similarity_search: str

class RAGEngine():

    def __init__(self, provider: LLMProvider, vector_store: InMemoryVectorStore):
        self.provider = provider
        self.vector_store = vector_store
        graph = self.build_graph()
        self.state_graph = graph
        load_dotenv()
        api_key = os.getenv("LANGCHAIN_API_KEY") 
        self.langsmith_client = Client(api_key)

    def retrieve(self, state: InputState) -> OverallState:
        """Retrieve relevant documents from the vector store using the incoming state.

        The graph will pass the input state (which contains the `question` key).
        Use that value to perform a similarity search and return a partial state
        containing the `context` key.
        """
        query = state.get("question")
        if not query:
            raise ValueError("similarity_search must be a non-empty string")

        # Default amount is 4 documents retrieved
        retrieved_docs = self.vector_store.similarity_search(query)
        return {"context": retrieved_docs}
    
    def generate(self, state: OverallState) -> OverallState:
        """Generate an answer using the retrieved context and the provider.

        The node receives the current overall state (which includes `question` and
        `context`). We build a prompt from the retrieved context and ask the LLM
        provider for an answer. The function returns a partial state with the
        `answer` key.
        """
        docs = state.get("context") or []
        question = state.get("question")

        docs_content = "\n\n".join([getattr(doc, "page_content", str(doc)) for doc in docs])

        # Pull a prompt template from LangSmith (if available) and invoke it to
        # produce the final prompt/messages to send to the LLM provider.
        prompt = None
        if self.langsmith_client is not None:
            try:
                prompt = self.langsmith_client.pull_prompt("rlm-eu/rag-prompt")
            except Exception:
                prompt = None

        if prompt is not None:
            try:
                messages = prompt.invoke({"question": question, "context": docs_content})
            except Exception:
                # fall back to simple concatenated prompt
                messages = f"Context:\n{docs_content}\n\nQuestion: {question}"
        else:
            messages = f"Context:\n{docs_content}\n\nQuestion: {question}"

        response = self.provider.query(messages)

        # Normalize the provider response to a string answer.
        if hasattr(response, "content"):
            answer_text = response.content
        elif isinstance(response, str):
            answer_text = response
        elif hasattr(response, "text"):
            answer_text = response.text
        else:
            answer_text = str(response)

        return {"answer": answer_text}
    
    def build_graph(self) -> StateGraph:
        builder = StateGraph(OverallState, input_schema=InputState, output_schema=OverallState)

        builder.add_node("retrieve", self.retrieve)
        builder.add_node("generate", self.generate)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        return builder.compile()
    
    def run_graph(self, state: InputState) -> dict:
        """Run the compiled state graph with the provided question and return the result.

        Returns the resulting state (preferably containing the `answer` key). This
        method is defensive about the returned type and prints a short summary to
        the console.
        """
        if not state["question"]:
            raise ValueError("question must be a non-empty string")

        # Invoke the compiled graph. The compiled graph implements a Runnable-like
        # interface and typically returns a dict-like state.
        result = self.state_graph.invoke({"question": state["question"]})

        # Safely extract keys from result (dict-like or other). Then format.
        context = None
        answer = None
        if isinstance(result, dict):
            context = result.get("context")
            answer = result.get("answer")
        else:
            # best-effort conversion to dict
            try:
                result_dict = dict(result)
                context = result_dict.get("context")
                answer = result_dict.get("answer")
            except Exception:
                # fallback: try attributes
                context = getattr(result, "context", None)
                answer = getattr(result, "answer", None)

        print("\n=== Question ===")
        print(state["question"])

        # Print a nicely formatted context and answer to the terminal.
        print("\n=== Retrieved context ===\n")
        self._print_context(context)

        print("\n=== Answer ===\n")
        self._print_answer(answer)

        return result

    # -- formatting helpers -------------------------------------------------
    def _print_context(self, docs):
        """Pretty-print a list of Document objects (or similar) to the terminal.

        Shows a short source (if available) and a short snippet of content for
        each retrieved chunk.
        """
        if not docs:
            print("(no retrieved documents)")
            return

        width = shutil.get_terminal_size((80, 20)).columns
        for i, doc in enumerate(docs, start=1):
            # document may be a mapping or an object with attributes
            meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})
            source = None
            if isinstance(meta, dict):
                source = meta.get("source") or meta.get("path") or meta.get("file")
            if not source:
                source = getattr(doc, "source", None)

            # If source is a full path, show only the basename (omit full location)
            if isinstance(source, str) and os.path.sep in source:
                try:
                    source = os.path.basename(source)
                except Exception:
                    source = "(file)"

            content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)

            print(f"[{i}] {source or 'unknown source'}")
            print("-" * width)

            # Print the entire content (not shortened). Wrap lines for terminal width.
            for paragraph in content.split("\n\n"):
                for line in paragraph.splitlines():
                    print(textwrap.fill(line, width=width))
                print()

            print("=" * width)

    def _print_answer(self, answer):
        """Pretty-print the model answer while omitting common metadata keys.

        Accepts strings, dicts, or objects. If the answer is a JSON string, it's
        parsed and printed in a readable way. Metadata keys like 'created',
        'model', 'id', 'object', and 'usage' are filtered out.
        """
        #TODO: improve answer printing 

        # regex to capture the content inside Message(content=' ... ')
        match = re.search(r"Message\(content='(.*?)',\s*role=", answer, re.DOTALL)

        if match:
            model_response = match.group(1)
            # optional: unescape literal \n to real newlines
            model_response = model_response.encode('utf-8').decode('unicode_escape')
            print(model_response.strip())
        else:
            print("No model content found.")
