from providers.factory import LLMProviderFactory
from providers.base_provider import Provider
from providers.huggingface_provider import HuggingFaceAPIProvider
from rag.document_loader import DocumentLoader
from rag.document_chunker import DocumentChunker
from rag.document_embedder import DocumentEmbedder
from pathlib import Path
import asyncio

def basic_openai_query():
    provider_factory = LLMProviderFactory()
    provider_factory.set_provider(Provider.OPENAI)
    provider_factory.set_model("gpt-4o")

    provider = provider_factory.get_provider()

    provider.query("Hello, how are you?")


def basic_huggingface_query():
    provider_factory = LLMProviderFactory()
    provider_factory.set_provider(Provider.HUGGINGFACE)
    provider_factory.set_model("deepseek-ai/DeepSeek-R1")

    provider : HuggingFaceAPIProvider  = provider_factory.get_provider()
    provider.set_huggingface_provider("together")

    response = provider.query("Hello, what are you?")
    print(f"HuggingFace Response: {response}")

def document_indexing(file_name: str):
    loader = DocumentLoader()
    pdf_path = Path(__file__).parent.parent / "papers" / file_name
    pages = asyncio.run(loader.load_pdf(pdf_path))

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(pages)

    embedder = DocumentEmbedder()
    embedder.set_embedding_model(Provider.HUGGINGFACE, "sentence-transformers/all-mpnet-base-v2")
    embedding_ids = embedder.embed_documents(chunks)

    print(f"Embedded {len(embedding_ids)} document chunks. First three embeddings are: {embedding_ids[:3]}" )


if __name__ == "__main__":
    file_name = "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf"

    document_indexing(file_name)
