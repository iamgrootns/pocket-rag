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

- Embedder + reranker download automatically from Hugging Face on first run
- The LLM does not auto-download. Get `qwen2.5-3b-instruct-q4_k_m.gguf`
  and drop it in `backend/Models/` (name configurable in `backend/config.py`)

### 2. Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux: source venv/bin/activate
pip install -r requirements.txt
python main.py           # FastAPI at 127.0.0.1:8765
```

### 3. Frontend (second terminal)

```bash
cd electron
npm install
npm start
```

The Electron app starts with the window; the backend must be running.

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
