from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    
    async def load_pdf(self, file_path: str) -> list:
        """Asynchronously loads and returns pages from a PDF file."""

        loader = PyPDFLoader(file_path)
        pages = []

        async for page in loader.alazy_load():
            pages.append(page)

        return pages