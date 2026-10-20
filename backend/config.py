"""
Global config for Pocket RAG.
Change values here only - no other file should hardcode settings.
"""

import os

# Base folder = backend/ folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All app data (db, uploads) lives under backend/data/
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "Models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# SQLite db file (main + vector store in one file)
DB_PATH = os.path.join(DATA_DIR, "ragdb.sqlite")

# --- Embedding model (tiny, good accuracy) ---
# bge-small-en-v1.5 = 384 dim, ~100MB, great for 4GB GPU
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# --- Reranker (cross-encoder, the big accuracy boost) ---
# ~500MB, runs fine on 4GB GPU / CPU if needed
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
# A snippet scores below this = weak evidence -> no answer
RERANKER_THRESHOLD = 0.35

# --- LLM (GGUF, Q4, runs in ~2GB VRAM) ---
# Auto-downloaded from Hugging Face on first run (like DFAPI models).
# Manual option: drop the .gguf into backend/Models/ and it is used as is.
LLM_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LLM_FILE_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_MODEL_PATH = os.environ.get(
    "RAG_LLM_MODEL",
    os.path.join(MODELS_DIR, LLM_FILE_NAME),
)
# How many top snippets are shown to the LLM
LLM_MAX_SNIPPETS = 8
# Conservative generation settings for grounded answers
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.2

# --- Retrieval settings ---
# How many results BM25 and vector search each return
RETRIEVAL_TOP_K = 30
# Final reranked snippets kept for generation
FINAL_TOP_K = 8
# RRF constant (standard value, trade-off between the two lists)
RRF_K = 60

# --- Chunking settings ---
# Parent chunks = big context pieces fed to the LLM
PARENT_CHUNK_SIZE = 1200  # chars
# Child chunks = small retrievable pieces
CHILD_CHUNK_SIZE = 400    # chars
CHILD_CHUNK_OVERLAP = 50  # chars

# --- Allowed document types ---
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Backend port the Electron app talks to
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8765
