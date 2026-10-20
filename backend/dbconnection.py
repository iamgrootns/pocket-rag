"""
Database connection management.
One file, one concern: only how SQLite connections are created.
"""

import sqlite3

from config import DB_PATH, EMBEDDING_DIM


class dbconnection:
    """Creates tuned sqlite connections and loads the vector extension."""

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # WAL = parallel reads while background process writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            # sqlite-vec = vector search inside this same db file
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            # fallback: app still works, embedding search degrades at runtime
            pass
        return conn

    def create_tables(self):
        """Tables: documents -> parent chunks -> child chunks (vectors)."""
        conn = self._connect()
        cur = conn.cursor()

        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_guid      TEXT PRIMARY KEY,
                filename      TEXT NOT NULL,
                file_hash     TEXT UNIQUE NOT NULL,
                page_count    INTEGER DEFAULT 0,
                status        TEXT DEFAULT 'Processing',
                created_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                chunk_guid UNINDEXED,
                doc_guid UNINDEXED,
                kind UNINDEXED,
                page_num UNINDEXED,
                content
            );

            CREATE TABLE IF NOT EXISTS child_meta (
                chunk_guid TEXT PRIMARY KEY,
                parent_guid TEXT NOT NULL,
                doc_guid TEXT NOT NULL,
                page_num INTEGER
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
                chunk_guid TEXT PRIMARY KEY,
                embedding float[%d]
            );
            """
            % EMBEDDING_DIM
        )

        conn.commit()
        conn.close()
