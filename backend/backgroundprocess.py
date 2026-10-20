"""
The Manager - all business logic in numbered steps.
Same pattern as DFAPI BackgroundProcessor:
endpoints hand work over, and the Manager calls specialist
functions step-by-step. Change a step = change it in its
specialist file once, and every pipeline keeps working.
"""

import json
import time

from config import RERANKER_THRESHOLD
from dbops import RagDBOperations
from chunking_util import ChunkingUtils
from commonfunc import CommonFunctions
from embedding_wrapper import EmbeddingWrapper
from ingestion_util import IngestionUtils
from llm_wrapper import LLMWrapper
from loggerutil import logger_util
from reranker_wrapper import RerankerWrapper
from retriever_util import RetrievalUtils
from state import processing_files


class BackgroundProcessor:
    """Central orchestrator. Built once in lifespan, reused by every endpoint."""

    def __init__(
        self,
        db_ops: RagDBOperations,
        commonfunc: CommonFunctions,
        ingestion: IngestionUtils,
        chunker: ChunkingUtils,
        embedder: EmbeddingWrapper,
        retriever: RetrievalUtils,
        reranker: RerankerWrapper,
        llm: LLMWrapper,
    ):
        self.db_ops = db_ops
        self.commonfunc = commonfunc
        self.ingestion = ingestion
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.logger = logger_util("manager")
        # guid -> filename cache so citations print real names
        self._doc_name_cache: dict[str, str] = {}

    # ------------------------------------------------------
    # PIPELINE 1: INGEST a document
    # ------------------------------------------------------
    async def process_ingest(self, file_bytes: bytes, doc_guid: str, filename: str):
        """Ingest pipeline: validate -> dedupe -> extract -> chunk -> embed -> store."""
        try:
            self.logger.info(f"[MG] Ingest started guid={doc_guid} file={filename}")

            # 1) file type must be allowed
            if not self.commonfunc.validate_file_type(filename):
                self.logger.info(f"[MG] Rejected (type) guid={doc_guid}")
                return {**self._intro(doc_guid, filename), "status": "Rejected: unsupported type"}

            # 2) duplicate check by file hash (same bytes already indexed)
            file_hash = self.commonfunc.compute_file_hash(file_bytes)
            existing = self.db_ops.find_doc_by_hash(file_hash)
            if existing:
                self.logger.info(f"[MG] Duplicate guid={doc_guid} -> {existing['doc_guid']}")
                return {
                    **self._intro(existing["doc_guid"], existing["filename"]),
                    "status": existing["status"],
                    "note": "Duplicate document - already in the library.",
                }

            # 3) extract text page by page
            pages = self.ingestion.extract_pages(file_bytes, filename)
            if not pages:
                return {**self._intro(doc_guid, filename), "status": "Failed: no extractable text"}

            # 4) chunk every page into parent -> children groups
            all_groups = []
            for page in pages:
                all_groups.extend(self.chunker.make_parent_children(page["text"], page["page_num"]))
            self.logger.info(f"[MG] Chunked guid={doc_guid} groups={len(all_groups)}")

            # 5) embed every child chunk in one batch
            all_children = [c for g in all_groups for c in g["children"]]
            vectors = self.embedder.embed_passages([c["content"] for c in all_children])
            for child, vector in zip(all_children, vectors):
                child["vector"] = vector
            self.logger.info(f"[MG] Embedded guid={doc_guid} children={len(all_children)}")

            # 6) store the document row and all chunk groups
            self.db_ops.insert_document(filename, file_hash, doc_guid)
            for group in all_groups:
                self.db_ops.insert_chunk_group(doc_guid, group["parent"], group["children"])

            # 7) mark complete and keep a copy of the original file
            self.db_ops.update_document(doc_guid, "Complete", page_count=len(pages))
            self.commonfunc.save_upload(file_bytes, filename, doc_guid)

            self.logger.info(f"[MG] Ingest finished guid={doc_guid}")
            return {
                **self._intro(doc_guid, filename),
                "status": "Complete",
                "pages": len(pages),
                "parents": len(all_groups),
                "children": len(all_children),
            }

        except Exception as e:
            self.logger.exception(f"[MG] Ingest failed guid={doc_guid}: {e}")
            return {"doc_guid": doc_guid, "filename": filename, "status": f"ERR: {e}"}
        finally:
            processing_files.discard(doc_guid)

    # ------------------------------------------------------
    # PIPELINE 2: QUERY (hybrid retrieve -> rerank -> grounded answer)
    # ------------------------------------------------------
    async def process_query(self, query_guid: str, question: str) -> dict:
        """Query pipeline: embed -> hybrid search -> rerank -> grounded LLM answer."""
        try:
            start = time.time()
            self.logger.info(f"[MG] Query started guid={query_guid} q={question!r}")

            # 1) hybrid retrieval: BM25 + vector search fused with RRF
            candidates = self.retriever.retrieve(question)
            if not candidates:
                return self._no_evidence_response(query_guid, question, start)

            # 2) child snippets -> their big parent chunks (more context for the LLM)
            parents = []
            seen_parents = set()
            for candidate in candidates:
                if candidate["parent_guid"] in seen_parents:
                    continue
                seen_parents.add(candidate["parent_guid"])
                parent = self.db_ops.fetch_parent(candidate["parent_guid"])
                if parent:
                    parents.append(
                        {
                            "parent_guid": candidate["parent_guid"],
                            "doc_guid": candidate["doc_guid"],
                            "page_num": parent["page_num"],
                            "content": parent["content"],
                        }
                    )

            # 3) cross-encoder rerank on the parents
            scores = self.reranker.rerank(question, [p["content"] for p in parents])
            for parent, score in zip(parents, scores):
                parent["rerank_score"] = float(score)

            # 4) keep only confident evidence, best first
            strong = sorted(
                [p for p in parents if p["rerank_score"] >= RERANKER_THRESHOLD],
                key=lambda p: p["rerank_score"],
                reverse=True,
            )
            if not strong:
                return self._no_evidence_response(query_guid, question, start)
            strong = strong[:8]

            # 5) attach filenames so citations read content.pdf p3
            for parent in strong:
                parent["source"] = self._doc_name(parent["doc_guid"])

            # 6) grounded generation - LLM gets numbered snippets only
            numbered = [
                {
                    "number": i + 1,
                    "content": p["content"],
                    "source": f"{p['source']} p{p['page_num']}",
                }
                for i, p in enumerate(strong)
            ]
            answer = self.llm.generate_grounded_answer(question, numbered)

            # 7) build the sources panel + save the answer
            sources = [
                {
                    "number": i + 1,
                    "source": p["source"],
                    "page": p["page_num"],
                    "parent_guid": p["parent_guid"],
                    "snippet": p["content"][:400],
                    "rerank_score": round(p["rerank_score"], 4),
                }
                for i, p in enumerate(strong)
            ]
            elapsed = int((time.time() - start) * 1000)
            self.db_ops.save_answer(query_guid, question, answer, json.dumps(sources))

            self.logger.info(f"[MG] Query finished guid={query_guid} ms={elapsed}")
            return {
                "query_guid": query_guid,
                "question": question,
                "answer": answer,
                "sources": sources,
                "processing_time_ms": elapsed,
            }

        except Exception as e:
            self.logger.exception(f"[MG] Query failed guid={query_guid}: {e}")
            return {"query_guid": query_guid, "status": f"ERR: {e}"}
        finally:
            processing_files.discard(query_guid)

    # ------------------------------------------------------
    # helpers (each tiny and focused)
    # ------------------------------------------------------
    @staticmethod
    def _intro(doc_guid: str, filename: str) -> dict:
        return {"doc_guid": doc_guid, "filename": filename}

    def _doc_name(self, doc_guid: str) -> str:
        """Cached doc_guid -> filename lookup."""
        if doc_guid not in self._doc_name_cache:
            doc = self.db_ops.get_document(doc_guid) or {}
            self._doc_name_cache[doc_guid] = doc.get("filename", "document")
        return self._doc_name_cache[doc_guid]

    def _no_evidence_response(self, query_guid: str, question: str, start: float) -> dict:
        self.logger.info(f"[MG] No strong evidence guid={query_guid}")
        return {
            "query_guid": query_guid,
            "question": question,
            "answer": "This is not covered in your documents.",
            "sources": [],
            "processing_time_ms": int((time.time() - start) * 1000),
        }
