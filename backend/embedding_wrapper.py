"""
Embedding model wrapper (external = wrapper class, uniform interface).
One class, one concern: text -> vectors. Model can be swapped here only.
"""

import torch
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME
from loggerutil import logger_util


class EmbeddingWrapper:
    """Turns text into dense vectors with bge-small."""

    def __init__(self):
        self.logger = logger_util("embedding")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Loading embedding model on {device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    def embed_query(self, query: str) -> list[float]:
        """Single query vector (bge needs an instruction prefix)."""
        vector = self.model.encode(
            f"Represent this sentence for searching relevant passages: {query}",
            normalize_embeddings=True,
        )
        return vector.tolist()

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        """Batch passage vectors (normalized so cosine works)."""
        vectors = self.model.encode(
            passages,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
