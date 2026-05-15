from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routes import diagnose, ingest, query, status
from rag.db import ensure_indexes, ensure_jobs_table, get_vectorstore
from rag.observability import get_logger, setup_logging
from rag.providers import get_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger = get_logger(__name__)

    provider = get_provider()
    embeddings = provider.get_embeddings()
    llm = provider.get_llm()
    vectorstore = get_vectorstore(embeddings, collection_name=provider.embed_model_name)

    # Schema migrations: idempotent. PGVector creates its own tables on
    # first use, so this runs after get_vectorstore to ensure the
    # embedding table exists before we try to index it.
    ensure_jobs_table()
    ensure_indexes()

    app.state.provider = provider
    app.state.llm = llm
    app.state.vectorstore = vectorstore

    logger.info(
        "app_ready",
        extra={
            "event": "app_ready",
            "provider": provider.__class__.__name__,
            "embed_model": provider.embed_model_name,
        },
    )

    yield


app = FastAPI(title="AI RAG System", lifespan=lifespan)

app.include_router(diagnose.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(status.router)


@app.get("/health")
def health():
    return {"status": "ok"}
