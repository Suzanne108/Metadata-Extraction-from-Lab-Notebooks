from providers.base_provider import Provider
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.vectorstores import InMemoryVectorStore

class DocumentEmbedder():

    def __init__(self):
        self.embedding_model = None
        self.vector_store = None


    def set_embedding_model(self, provider: Provider, model: str):
        """Sets the embedding model to be used and creates the vector store"""

        if provider == Provider.OPENAI:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")

            self.embedding_model = OpenAIEmbeddings(model=model)
            self.vector_store = InMemoryVectorStore(self.embedding_model)
        elif provider == Provider.HUGGINGFACE:
            load_dotenv()
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            
            if api_key is None:
                raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")

            self.embedding_model = HuggingFaceEmbeddings(model=model)
            self.vector_store = InMemoryVectorStore(self.embedding_model)

        else:
            raise ValueError("Unsupported provider")
                             
    
    def embed_documents(self, chunks: list):
        """Embeds and adds document chunks to the vector store""" 

        if self.vector_store is not None:
            chunk_ids = self.vector_store.add_documents(chunks)

            return chunk_ids
        else:
            raise ValueError("Vector store is not initialized.")

