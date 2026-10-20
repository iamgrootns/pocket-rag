"""
Retrieval: BM25 + vector search fused with Reciprocal Rank Fusion.
One class, one concern: given a query -> best parent snippets.
"""

from config import RETRIEVAL_TOP_K, RRF_K
from loggerutil import logger_util


class RetrievalUtils:
    """Hybrid retriever. Needs db_ops + embedding wrapper (passed by Manager)."""

    def __init__(self, db_ops, embedding_wrapper):
        self.db_ops = db_ops
        self.embedder = embedding_wrapper
        self.logger = logger_util("retrieval")

    def retrieve(self, query: str) -> list[dict]:
        """Returns fused parent candidates: [{"parent_guid", "score", "child_contents"}]."""

        bm25_hits = self.db_ops.bm25_search(query, RETRIEVAL_TOP_K)
        query_vec = self.embedder.embed_query(query)
        vec_hits = self.db_ops.vector_search(query_vec, RETRIEVAL_TOP_K)

        # ---- reciprocal rank fusion ----
        # rank i in each list contributes 1 / (RRF_K + position)
        fused = {}

        for position, hit in enumerate(bm25_hits):
            fused[hit["chunk_guid"]] = {"first_rank": position}
            fused[hit["chunk_guid"]]["bm25"] = 1.0 / (RRF_K + position + 1)

        for position, (chunk_guid, _distance) in enumerate(vec_hits):
            if chunk_guid in fused:
                fused[chunk_guid]["vec"] = 1.0 / (RRF_K + position + 1)
            else:
                fused[chunk_guid] = {
                    "first_rank": RETRIEVAL_TOP_K + position,
                    "vec": 1.0 / (RRF_K + position + 1),
                }

        for chunk_guid, entry in fused.items():
            entry["score"] = entry.get("bm25", 0.0) + entry.get("vec", 0.0)

        ordered = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)

        candidates = []
        seen_parents = set()
        for chunk_guid, entry in ordered:
            meta = self.db_ops.get_parent_of_child(chunk_guid)
            if meta is None or meta["parent_guid"] in seen_parents:
                continue
            seen_parents.add(meta["parent_guid"])

            child = self.db_ops.fetch_child(chunk_guid)
            if child is None:
                continue
            candidates.append(
                {
                    "parent_guid": meta["parent_guid"],
                    "doc_guid": meta["doc_guid"],
                    "child_guid": chunk_guid,
                    "child_content": child["content"],
                    "score": entry["score"],
                }
            )
        return candidates
