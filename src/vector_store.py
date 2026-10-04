import os
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import VECTOR_STORE_DIR
from src.embeddings import get_embedding_model


class StudyVectorStore:
    """FAISS Vector Store Manager for indexing and retrieving study materials."""

    def __init__(
        self,
        embedding_model: Optional[Embeddings] = None,
        index_dir: str = str(VECTOR_STORE_DIR),
    ):
        self.embeddings = embedding_model or get_embedding_model()
        self.index_dir = Path(index_dir)
        self.vector_store: Optional[FAISS] = None

    def build_index(self, documents: List[Document]) -> FAISS:
        """Creates a FAISS vector store index from a list of documents."""
        if not documents:
            raise ValueError("No documents provided to build vector index.")

        print(f"Building FAISS vector index for {len(documents)} document chunks...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store

    def save_index(self, folder_name: str = "index") -> str:
        """Persists the FAISS index to disk."""
        if not self.vector_store:
            raise ValueError("No vector store index exists to save.")

        save_path = self.index_dir / folder_name
        self.vector_store.save_local(str(save_path))
        print(f"FAISS index saved successfully to: {save_path}")
        return str(save_path)

    def load_index(self, folder_name: str = "index") -> FAISS:
        """Loads a persisted FAISS index from disk."""
        load_path = self.index_dir / folder_name
        if not load_path.exists():
            raise FileNotFoundError(f"No FAISS index found at {load_path}")

        print(f"Loading FAISS index from: {load_path}")
        self.vector_store = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.vector_store

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Performs similarity search with relevance scores."""
        if not self.vector_store:
            raise ValueError("Vector store index is not loaded or initialized.")

        return self.vector_store.similarity_search_with_score(query, k=k)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Returns the FAISS store as a LangChain retriever."""
        if not self.vector_store:
            raise ValueError("Vector store index is not loaded or initialized.")

        kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=kwargs)
