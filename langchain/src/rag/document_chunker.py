from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DocumentChunker():
    
    def chunk_documents(self, documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Chunks documents into smaller pieces"""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        all_chunks = text_splitter.split_documents(documents)
        
        return all_chunks