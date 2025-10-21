from providers.factory import LLMProviderFactory
from providers.base_provider import Provider
from rag.document_loader import DocumentLoader
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

    print(f"{pages[0].metadata}\n")
    print(pages[0].page_content)