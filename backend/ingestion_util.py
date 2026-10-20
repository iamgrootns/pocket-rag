"""
Text extraction from documents.
One class, one concern: bytes -> clean text + per-chunk page numbers.
Change extraction strategy here and everything downstream updates.
"""

import io
import os

from loggerutil import logger_util

# document readers - optional, so the backend still boots without them
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import docx
except ImportError:  # pragma: no cover
    docx = None


class IngestionUtils:
    """Extracts page-aligned text from PDF/DOCX/TXT/MD."""

    def __init__(self):
        self.logger = logger_util("ingestion")

    def extract_pages(self, file_bytes: bytes, filename: str) -> list[dict]:
        """Returns list of {"page_num": int, "text": str}. page_num from 1."""
        ext = os.path.splitext(filename or "")[1].lower()

        if ext == ".pdf":
            return self._extract_pdf(file_bytes)
        if ext == ".docx":
            return self._extract_docx(file_bytes)
        # txt/md = whole file is page 1
        text = file_bytes.decode("utf-8", errors="ignore")
        if text.strip():
            return [{"page_num": 1, "text": text}]
        return []

    def _extract_pdf(self, file_bytes: bytes) -> list[dict]:
        try:
            if PdfReader is None:
                self.logger.error("pypdf is not installed - cannot read PDF files")
                return []

            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page_num": i, "text": text})
            return pages
        except Exception as e:
            self.logger.error(f"pdf extraction failed: {e}")
            return []

    def _extract_docx(self, file_bytes: bytes) -> list[dict]:
        try:
            if docx is None:
                self.logger.error("python-docx is not installed - cannot read DOCX files")
                return []

            document = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
            if text.strip():
                return [{"page_num": 1, "text": text}]
            return []
        except Exception as e:
            self.logger.error(f"docx extraction failed: {e}")
            return []
