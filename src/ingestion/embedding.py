import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# Suppress excessive logging, while keeping errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize module-level model singleton
_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Retrieve the cached MiniLM model, loading it if necessary.
    
    Returns:
        SentenceTransformer: A ready-to-use model instance.
    """
    
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def embed_text(text: str) -> np.ndarray:
    """
    Generate a 384-dimensional normalized embedding using MiniLM.

    Args:
        text (str): The text content to embed.
    
    Returns:
        np.ndarray: Normalized embedding of shape (384,) as float32.
    """

    model = get_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding.astype(np.float32)
