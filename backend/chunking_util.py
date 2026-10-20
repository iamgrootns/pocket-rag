"""
Parent-child chunking.
One class, one concern: page text -> parent chunks (context) -> child chunks (retrieval).
Change sizes / strategy here once - ingestion and query both benefit.
"""

import uuid

from config import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE
from loggerutil import logger_util


class ChunkingUtils:
    """Builds parent -> children chunk groups from extracted pages."""

    def __init__(self):
        self.logger = logger_util("chunking")

    def make_parent_children(self, page_text: str, page_num: int) -> list[dict]:
        """One page -> list of groups.

        Each group:
        parent  = ~PARENT_CHUNK_SIZE chars (fed to the LLM)
        children = ~CHILD_CHUNK_SIZE pieces, overlapping slightly (retrieval)
        """
        groups = []

        parents = self._split_text(page_text, PARENT_CHUNK_SIZE)

        for parent_text in parents:
            if not parent_text.strip():
                continue

            parent_guid = str(uuid.uuid4())
            children = []
            child_texts = self._split_text(parent_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)

            for child_text in child_texts:
                children.append(
                    {
                        "guid": str(uuid.uuid4()),
                        "page_num": page_num,
                        "content": child_text,
                    }
                )

            groups.append(
                {
                    "parent": {
                        "guid": parent_guid,
                        "page_num": page_num,
                        "content": parent_text,
                    },
                    "children": children,
                }
            )

        return groups

    def _split_text(self, text: str, size: int, overlap: int = 0) -> list[str]:
        """Slide a window over text, trying to cut at paragraph/sentence edges."""
        text = text.strip()
        if len(text) <= size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))

            # cut at the last paragraph break inside the window (cleaner chunks)
            cut = text.rfind("\n\n", start, end)
            if cut == -1:
                cut = text.rfind("\n", start, end)
            if cut == -1:
                cut = text.rfind(". ", start, end)
            if cut > start:
                end = cut + 1

            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

        return chunks
