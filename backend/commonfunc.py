"""
Common utilities: file validation, hashing, storage.
One class, one concern. Used by all pipelines via the Manager.
"""

import hashlib
import os

from config import ALLOWED_EXTENSIONS, UPLOADS_DIR
from loggerutil import logger_util


class CommonFunctions:
    """Small shared helpers every pipeline needs."""

    def __init__(self):
        self.logger = logger_util("commonfunc")

    def validate_file_type(self, filename: str) -> bool:
        """True only for .pdf/.docx/.txt/.md - one place to fix types."""
        ext = os.path.splitext(filename or "")[1].lower()
        return ext in ALLOWED_EXTENSIONS

    def compute_file_hash(self, file_bytes: bytes) -> str:
        """Stable sha256 - used for duplicate-document detection."""
        return hashlib.sha256(file_bytes).hexdigest()

    def save_upload(self, file_bytes: bytes, filename: str, doc_guid: str) -> str:
        """Save original bytes under data/uploads/<guid>_<name>. Returns path."""
        safe_name = os.path.basename(filename)
        path = os.path.join(UPLOADS_DIR, f"{doc_guid}_{safe_name}")
        with open(path, "wb") as f:
            f.write(file_bytes)
        return path
