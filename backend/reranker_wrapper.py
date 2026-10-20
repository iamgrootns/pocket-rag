"""
Reranker wrapper (external = wrapper class, uniform interface).
One class, one concern: (query, snippets) -> relevance scores in [0,1].
The single biggest accuracy lever in the whole pipeline.
"""

import torch
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME
from loggerutil import logger_util


class RerankerWrapper:
    """Cross-encoder reranking with bge-reranker-base."""

    def __init__(self):
        self.logger = logger_util("reranker")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Loading reranker on {device}")
        self.model = CrossEncoder(RERANKER_MODEL_NAME, device=device)

    def rerank(self, query: str, snippets: list[str]) -> list[float]:
        """Score every snippet against the query in one batch."""
        if not snippets:
            return []
        scores = self.model.predict(
            [(query, snippet) for snippet in snippets],
            show_progress_bar=False,
        )
        return scores.tolist()
