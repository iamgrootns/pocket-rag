"""
All DB write/read operations for Pocket RAG.
One class, one concern: only database ops. Manager calls these.
"""

import json
import sqlite3
import uuid

from dbconnection import dbconnection
from loggerutil import logger_util


class RagDBOperations:
    """SQLite document / chunk / answer storage."""

    def __init__(self):
        self.dbconn = dbconnection()
        self.logger = logger_util("dbops")
        self.dbconn.create_tables()

    # ---------- documents ----------

    def insert_document(self, filename: str, file_hash: str, doc_guid: str):

        conn = self.dbconn._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO documents (doc_guid, filename, file_hash, status) VALUES (?, ?, ?, 'Processing')",
                (doc_guid, filename, file_hash),
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            self.logger.error(f"insert_document failed: {e}")
            return False
        finally:
            conn.close()

    def find_doc_by_hash(self, file_hash: str):
        conn = self.dbconn._connect()
        row = conn.execute(
            "SELECT doc_guid, filename, status FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def update_document(self, doc_guid: str, status: str, page_count: int = 0):
        conn = self.dbconn._connect()
        try:
            conn.execute(
                "UPDATE documents SET status = ?, page_count = ? WHERE doc_guid = ?",
                (status, page_count, doc_guid),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_document(self, doc_guid: str):
        conn = self.dbconn._connect()
        row = conn.execute(
            "SELECT doc_guid, filename, status, page_count, created_at FROM documents WHERE doc_guid = ?",
            (doc_guid,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_documents(self):
        conn = self.dbconn._connect()
        rows = conn.execute(
            "SELECT doc_guid, filename, status, page_count, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_document(self, doc_guid: str) -> bool:
        """Remove doc + its chunks and vectors."""
        try:
            conn = self.dbconn._connect()
            conn.execute("DELETE FROM child_meta WHERE doc_guid = ?", (doc_guid,))
            conn.execute("DELETE FROM chunks WHERE doc_guid = ?", (doc_guid,))
            vector_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT chunk_guid FROM child_meta WHERE doc_guid = ?", (doc_guid,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM documents WHERE doc_guid = ?", (doc_guid,))
            conn.commit()
            conn.close()
            # vectors live in a virtual table - delete with separate connection
            vconn = self.dbconn._connect()
            for cid in vector_ids:
                vconn.execute("DELETE FROM vectors WHERE chunk_guid = ?", (cid,))
            vconn.commit()
            vconn.close()
            return True
        except Exception as e:
            self.logger.error(f"delete_document failed: {e}")
            return False

    # ---------- chunks ----------

    def insert_chunk_group(self, doc_guid: str, parent, children):
        """Insert one parent chunk + its child chunks.

        parent = {"guid", "page_num", "content"}
        children = list of {"guid", "page_num", "content", "vector"}
        """
        conn = self.dbconn._connect()
        try:
            conn.execute(
                "INSERT INTO chunks (chunk_guid, doc_guid, kind, page_num, content) VALUES (?,?,?,?,?)",
                (parent["guid"], doc_guid, "parent", parent["page_num"], parent["content"]),
            )
            for child in children:
                conn.execute(
                    "INSERT INTO chunks (chunk_guid, doc_guid, kind, page_num, content) VALUES (?,?,?,?,?)",
                    (child["guid"], doc_guid, "child", child["page_num"], child["content"]),
                )
                conn.execute(
                    "INSERT INTO child_meta (chunk_guid, parent_guid, doc_guid, page_num) VALUES (?,?,?,?)",
                    (child["guid"], parent["guid"], doc_guid, child["page_num"]),
                )
            conn.commit()

            # embeddings go into the virtual vec table
            vconn = self.dbconn._connect()
            for child in children:
                vconn.execute(
                    "INSERT INTO vectors (chunk_guid, embedding) VALUES (?, ?)",
                    (child["guid"], json.dumps(child["vector"])),
                )
            vconn.commit()
            vconn.close()
            return True
        except Exception as e:
            self.logger.error(f"insert_chunk_group failed: {e}")
            return False
        finally:
            conn.close()

    # ---------- retrieval ----------

    def bm25_search(self, query: str, top_k: int):
        """Full-text keyword search using fts5 bm25 ranking."""
        conn = self.dbconn._connect()
        rows = conn.execute(
            """
            SELECT chunk_guid, content, page_num, bm25(chunks) AS rank
            FROM chunks WHERE chunks MATCH ? AND kind = 'child'
            ORDER BY rank LIMIT ?
            """,
            (query, top_k),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def vector_search(self, query_vector, top_k: int):
        """Nearest child chunks by cosine-ish distance (sqlite-vec)."""
        vconn = self.dbconn._connect()
        try:
            rows = vconn.execute(
                """
                SELECT chunk_guid, distance FROM vectors
                WHERE embedding MATCH ? AND k = ?
                """,
                (json.dumps(list(query_vector)), top_k),
            ).fetchall()
            results = [(r[0], r[1]) for r in rows]
        except Exception as e:
            self.logger.error(f"vector_search failed: {e}")
            results = []
        finally:
            vconn.close()
        return results

    def fetch_parent(self, parent_guid: str):
        conn = self.dbconn._connect()
        row = conn.execute(
            "SELECT content, page_num FROM chunks WHERE chunk_guid = ? AND kind = 'parent'",
            (parent_guid,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def fetch_child(self, chunk_guid: str):
        conn = self.dbconn._connect()
        row = conn.execute(
            "SELECT content, page_num FROM chunks WHERE chunk_guid = ?",
            (chunk_guid,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_parent_of_child(self, chunk_guid: str):
        conn = self.dbconn._connect()
        row = conn.execute(
            "SELECT parent_guid, doc_guid FROM child_meta WHERE chunk_guid = ?", (chunk_guid,)
        ).fetchone()
        conn.close()
        return {"parent_guid": row[0], "doc_guid": row[1]} if row else None

    # ---------- answers ----------

    def save_answer(self, query_guid: str, question: str, answer: str, sources_json: str):
        conn = self.dbconn._connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS answers (query_guid TEXT PRIMARY KEY, question TEXT, answer TEXT, sources TEXT, created_at TEXT DEFAULT (datetime('now')))"
            )
            conn.execute(
                "INSERT INTO answers (query_guid, question, answer, sources) VALUES (?,?,?,?)",
                (query_guid, question, answer, sources_json),
            )
            conn.commit()
        finally:
            conn.close()

    def get_answer(self, query_guid: str):
        conn = self.dbconn._connect()
        row = conn.execute(
            "SELECT query_guid, question, answer, sources, created_at FROM answers WHERE query_guid = ?",
            (query_guid,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        return d
