from providers.factory import LLMProviderFactory
from providers.base_provider import Provider
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

if __name__ == "__main__":
    loader = DocumentLoader()
    pdf_path = Path(__file__).parent.parent / "papers" / "Image-based_methods_for_phenotyping_growth_dynamics_and_fitness_components_in_arabidopsis_thaliana.pdf"
    pages = asyncio.run(loader.load_pdf(pdf_path))

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(pages)

    embedder = DocumentEmbedder()
    embedder.set_embedding_model(Provider.OPENAI, "text-embedding-3-small")
    embedding_ids = embedder.embed_documents(chunks)

    print(f"Embedded {len(embedding_ids)} document chunks. First three embeddings are: {embedding_ids[:3]}" )