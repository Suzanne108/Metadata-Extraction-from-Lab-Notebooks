from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.model import get_embedding_model
from langchain_core.vectorstores import InMemoryVectorStore

class DocumentManager():

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = InMemoryVectorStore(embedding=self.embedding_model)
        self.document = None

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF into LangChain document objects."""

        loader = PyPDFLoader(file_path, mode="single")
        doc = loader.load()
        self.document = doc

        return doc

    def _chunk_doc(self, documents: List[Document], chunk_size: int = 1500, chunk_overlap: int = 150) -> List[Document]:
        """Split documents into overlapping chunks for embedding.

        Parameters:
            documents: The source `Document` objects to split.
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters shared between consecutive chunks.

        Returns:
            A list of chunked `Document` objects ready for embedding.
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            add_start_index=True
            )
    
        chunks = text_splitter.split_documents(documents)
        return chunks

    def embed_documents(self, file_path: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> list[str]:
        """Load, chunk, embed, and store a PDF in the vector store.

        Parameters:
            file_path: Path to the PDF file to ingest.
            chunk_size: Maximum number of characters per chunk passed to the splitter.
            chunk_overlap: Number of overlapping characters between adjacent chunks.

        Returns:
            The list of vector store document IDs returned by `add_documents`.
        """

        docs = self.load_pdf(file_path)
        chunks = self._chunk_doc(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_ids = self.vector_store.add_documents(chunks)

        return chunk_ids
    
    def retrieve_semantic_chunk(self, query: str, k: int = 4):
        """Retrieve relevant chunks from the vector store for a query.

        Parameters:
            query: The search text used for similarity search.
            k: The number of matching chunks to return.

        Returns:
            A list of matching chunk texts ordered by similarity.
        """

        retrieved_docs = self.vector_store.similarity_search(query, k=k)

        retrieved_text = []
        for doc in retrieved_docs:
            retrieved_text.append(doc.page_content)

        return retrieved_text
    


