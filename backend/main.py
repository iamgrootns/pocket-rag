"""
FastAPI backend for Pocket RAG.
Thin endpoints only: validate -> guid -> hand off to Manager -> return guid.
All business logic lives in backgroundprocess.BackgroundProcessor.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from backgroundprocess import BackgroundProcessor
from chunking_util import ChunkingUtils
from commonfunc import CommonFunctions
from config import BACKEND_HOST, BACKEND_PORT
from dbops import RagDBOperations
from embedding_wrapper import EmbeddingWrapper
from ingestion_util import IngestionUtils
from llm_wrapper import LLMWrapper
from loggerutil import logger_util
from reranker_wrapper import RerankerWrapper
from retriever_util import RetrievalUtils
from state import MAX_CONCURRENT_TASKS, processing_files, processing_lock

logger = logger_util("main")

# global instances loaded in lifespan
background_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model exactly once at startup (same idea as DFAPI lifespan)."""
    global background_processor

    logger.info("Lifespan startup: loading pipeline")

    db_ops = RagDBOperations()
    commonfunc = CommonFunctions()
    ingestion = IngestionUtils()
    chunker = ChunkingUtils()
    embedder = EmbeddingWrapper()

    retriever = RetrievalUtils(db_ops=db_ops, embedding_wrapper=embedder)
    reranker = RerankerWrapper()
    llm = LLMWrapper()

    background_processor = BackgroundProcessor(
        db_ops=db_ops,
        commonfunc=commonfunc,
        ingestion=ingestion,
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )

    logger.info("Lifespan startup complete")

    yield

    logger.info("Lifespan shutdown")


app = FastAPI(
    title="Pocket RAG",
    version="0.1",
    description="Local grounded RAG - Electron app + Python backend",
    lifespan=lifespan,
)


@app.get("/")
async def health():
    logger.info("Health check")
    return {"status": "Pocket RAG backend running", "version": "0.1"}


@app.post("/ingest/", response_class=JSONResponse)
async def ingest(file: UploadFile):
    # 1) validate + 2) guid + 3) hand off to Manager - nothing else here
    if not file.filename:
        logger.info("[EP] ingest rejected: empty filename")
        return {"doc_guid": "", "status": "filename missing"}

    doc_guid = str(uuid.uuid4())

    with processing_lock:
        if len(processing_files) >= MAX_CONCURRENT_TASKS:
            return {"doc_guid": "", "status": "busy - try again in a moment"}
        processing_files.add(doc_guid)

    file_bytes = await file.read()

    def _run_ingest():
        asyncio.run(
            background_processor.process_ingest(file_bytes, doc_guid, file.filename)
        )

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_ingest)

    return {"doc_guid": doc_guid, "filename": file.filename, "status": "Processing"}


@app.post("/query/", response_class=JSONResponse)
async def query(question: str):
    # single request at a time (4GB GPU budget)
    with processing_lock:
        if len(processing_files) >= MAX_CONCURRENT_TASKS:
            return {"query_guid": "", "status": "busy - try again in a moment"}
        query_guid = str(uuid.uuid4())
        processing_files.add(query_guid)

    def _run_query():
        asyncio.run(background_processor.process_query(query_guid, question))

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_query)

    return {"query_guid": query_guid, "status": "Processing"}


@app.get("/answer/{query_guid}", response_class=JSONResponse)
async def answer(query_guid: str):
    return {"query_guid": query_guid, "result": background_processor.db_ops.get_answer(query_guid)}


@app.get("/ingest_status/{doc_guid}", response_class=JSONResponse)
async def ingest_status(doc_guid: str):
    return {"doc_guid": doc_guid, "result": background_processor.db_ops.get_document(doc_guid)}


@app.get("/documents", response_class=JSONResponse)
async def documents():
    return {"documents": background_processor.db_ops.list_documents()}


@app.delete("/documents/{doc_guid}", response_class=JSONResponse)
async def delete_document(doc_guid: str):
    ok = background_processor.db_ops.delete_document(doc_guid)
    return {"doc_guid": doc_guid, "deleted": ok}


if __name__ == "__main__":
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
