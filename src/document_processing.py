from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.model import get_embedding_model
from langchain_core.vectorstores import InMemoryVectorStore
from src.utils import log



class DocumentManager():

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = InMemoryVectorStore(embedding=self.embedding_model)
        self.document = None

    def load_pdf(self, file_path: str) -> List[Document]:
        """Loads and returns pages from a PDF file"""

        loader = PyPDFLoader(file_path, mode="single")
        doc = loader.load()

        self.document = doc

        return doc

    def _chunk_doc(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """ Chunks documents into smaller pieces """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            add_start_index=True
            )
    
        chunks = text_splitter.split_documents(documents)
        return chunks

    def embed_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Embeds and adds document to the vector store"""

        docs = self.load_pdf(file_path)
        chunks = self._chunk_doc(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_ids = self.vector_store.add_documents(chunks)

        log(f"Added document to vector store")

        return chunk_ids
    
    def retrieve_semantic_chunk(self, query: str, k: int = 4):
        """Retrieve relevant chunks from the documents added to the vector store given the query.

        query: str = the query to perform similarity search on
        k: int = the amount of chunks to retrieve
        """

        retrieved_docs = self.vector_store.similarity_search(query, k=k)

        retrieved_text = []
        for doc in retrieved_docs:
            retrieved_text.append(doc.page_content)

        return retrieved_text
    


