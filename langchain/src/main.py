from providers.factory import LLMProviderFactory
from providers.base_provider import Provider
from providers.huggingface_provider import HuggingFaceAPIProvider
from rag.document_loader import DocumentLoader
from rag.document_chunker import DocumentChunker
from rag.document_embedder import DocumentEmbedder
from rag.rag_engine import RAGEngine, OverallState, InputState
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
import asyncio


def basic_openai_query():
    provider_factory = LLMProviderFactory()
    provider_factory.set_provider(Provider.OPENAI)
    provider_factory.set_model("gpt-4o")

    provider = provider_factory.get_provider()

    provider.query("Hello, how are you?")


def basic_huggingface_query() -> InMemoryVectorStore:
    provider_factory = LLMProviderFactory()
    provider_factory.set_provider(Provider.HUGGINGFACE)
    provider_factory.set_model("deepseek-ai/DeepSeek-R1")

    provider : HuggingFaceAPIProvider  = provider_factory.get_provider()
    provider.set_huggingface_provider("together")

    response = provider.query("Hello, what are you?")
    print(f"HuggingFace Response: {response}")

def document_indexing(file_name: str):
    """Indexes documents from a PDF file, embeds them, and returns the vector store."""
    loader = DocumentLoader()
    pdf_path = Path(__file__).parent.parent / "papers" / file_name
    pages = asyncio.run(loader.load_pdf(pdf_path))

    print(f"Total pages loaded from PDF: {len(pages)}")

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(pages)

    print(f"Total document chunks created: {len(chunks)}")

    embedder = DocumentEmbedder()
    embedder.set_embedding_model(Provider.HUGGINGFACE, "sentence-transformers/all-mpnet-base-v2")
    embedding_ids = embedder.embed_documents(chunks)

    print(f"Embedded {len(embedding_ids)} document chunks. First three embeddings are: {embedding_ids[:3]}" )
    
    return embedder.vector_store

def run_rag_query(vector_store: InMemoryVectorStore, question: str, search: str):
    """Runs a RAG query using the provided vector store and question."""

    if search == "":
        search = question

    initial_state: InputState = {
        "question": question,
        "similarity_search": search
    }

    provider_factory = LLMProviderFactory()
    provider_factory.set_provider(Provider.HUGGINGFACE)
    provider_factory.set_model("deepseek-ai/DeepSeek-R1")

    provider : HuggingFaceAPIProvider  = provider_factory.get_provider()
    provider.set_huggingface_provider("together")

    rag_engine = RAGEngine(provider, vector_store)
    result = rag_engine.run_graph(initial_state)

    return result
    

if __name__ == "__main__":

    file_name = "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf"

    vector_store = document_indexing(file_name)
    run_rag_query(vector_store, "What is the main focus of the paper?", "")

    # Investigation
    run_rag_query(vector_store, "Considering the ISA framework, please provide me with information on the investigation of this research. Who are the people involved? What organizations are affiliated?", "")

    # Study
    run_rag_query(vector_store, "Considering the ISA framework, please provide me with information on the study of this research. What is the study design, what factors and protocls are involved?", "")

    # Assay
    run_rag_query(vector_store, "Considering the ISA framework, please provide me with information on the assays involved in this research. What materials are used and please descirbe the samples, technologies used and measurements taken", "")
