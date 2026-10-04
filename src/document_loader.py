import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR


class StudyDocumentLoader:
    """Utility class to load and chunk study materials (PDF, TXT, MD)."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_single_document(self, file_path: str) -> List[Document]:
        """Loads a single document based on file extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext in [".txt", ".log"]:
            loader = TextLoader(str(path), encoding="utf-8")
        elif ext == ".md":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        documents = loader.load()
        return self.splitter.split_documents(documents)

    def load_directory(self, dir_path: str = str(DOCS_DIR)) -> List[Document]:
        """Loads all supported document types from a directory."""
        path = Path(dir_path)
        if not path.exists():
            return []

        all_docs = []
        for file in path.glob("**/*"):
            if file.suffix.lower() in [".pdf", ".txt", ".md"]:
                try:
                    docs = self.load_single_document(str(file))
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading {file.name}: {e}")

        return all_docs
