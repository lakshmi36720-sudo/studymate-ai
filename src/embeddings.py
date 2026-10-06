from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.config import EMBEDDING_MODEL_NAME, OPENAI_API_KEY


def get_embedding_model(provider: str = "huggingface") -> Embeddings:
    """
    Initializes and returns an embedding model based on provider preference.
    Default is HuggingFace (runs locally without requiring API key).
    """
    if provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in environment or .env file.")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Default to HuggingFace Sentence Transformers
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
