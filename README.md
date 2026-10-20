# Pocket RAG

Local grounded document Q&A. Electron desktop app + Python FastAPI backend.
Everything runs on your machine - designed and sized for a **4GB GPU laptop**.

**Ask a question -> it retrieves, reranks, and answers only from your documents with numbered citations. No internet needed at runtime.**

## How it computes an answer (accuracy pipeline)

1. Ingest: PDF/DOCX/TXT/MD -> page-aligned text -> parent-child chunks
2. Retrieval: hybrid BM25 + dense embeddings, fused with Reciprocal Rank Fusion
3. Rerank: cross-encoder (bge-reranker-base) - the big accuracy lever
4. Abstain: below confidence threshold -> "This is not covered in your documents."
5. Generate: Qwen2.5-3B GGUF Q4, grounded prompt, citations like `[1]`

## Setup

### 1. Backend models

All models download automatically on first run (same behaviour as the deepfake API models):

- Embedder `BAAI/bge-small-en-v1.5` + reranker `BAAI/bge-reranker-base` - Hugging Face cache
- LLM `Qwen/Qwen2.5-3B-Instruct-GGUF` (`qwen2.5-3b-instruct-q4_k_m.gguf`) - saved in `backend/Models/`

Offline? Put the GGUF into `backend/Models/` yourself and it is used as is.

### 2. Backend

```bash
cd backend
uv sync                # creates .venv with Python 3.10 + all deps (CUDA wheels)
uv run python main.py  # FastAPI at 127.0.0.1:8765
```

`uv sync` does everything: pinned Python 3.10 (via `.python-version`), a virtual
env in `backend/.venv`, and the CUDA builds of torch + llama-cpp-python from
prebuilt wheels (no local C++ compile).

### 3. Frontend (second terminal)

```bash
cd electron
npm install
npm start
```

The Electron app starts with the window; the backend must be running.

## Everyday commands

```bash
cd backend
uv sync                 # re-sync deps after editing pyproject.toml
uv run python main.py   # start the backend
uv add <package>        # add a dependency
uv run python -c "import torch; print(torch.cuda.is_available())"   # check GPU
```

## Layout

```
backend/
  main.py                  thin endpoints only
  backgroundprocess.py     Manager - all pipelines, numbered steps
  ingestion_util.py        bytes -> page-aligned text (one class)
  chunking_util.py         parent-child chunking (one class)
  embedding_wrapper.py     embeddings (swap models here)
  retriever_util.py        BM25 + vector + RRF fusion
  reranker_wrapper.py      cross-encoder reranking
  llm_wrapper.py           grounded generation
  dbops.py                 all SQLite operations
  dbconnection.py          connection handling + schema
  commonfunc.py            validation, hashing, uploads
  config.py                every tuning knob in one file
electron/
  main.js, preload.js, src/  Desktop app
```

## Tuning for your machine

All knobs live in `backend/config.py` (chunk sizes, top-k, rerank threshold).
If your GPU runs out of memory, set `RAG_GPU_LAYERS=0` to keep the LLM on CPU
(embeddings and reranker stay on GPU/CPU automatically).

## License

MIT
