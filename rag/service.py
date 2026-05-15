import time

from rag.chat import generate_answer
from rag.observability import get_logger
from rag.retrieve import get_retriever


logger = get_logger(__name__)


async def answer_query(
    *,
    question: str,
    k: int,
    filter: dict | None,
    vectorstore,
    llm,
    provider_name: str,
    embed_model: str,
) -> dict:
    """End-to-end query: retrieve → log → generate → return answer + sources.

    Designed to be the single seam between the HTTP layer and retrieval
    so future tools (NL→SQL, log search, code search) can wrap or replace
    pieces of this without touching FastAPI routes.
    """
    retriever = get_retriever(vectorstore, k=k, filter=filter)

    t0 = time.perf_counter()
    docs = await retriever.ainvoke(question)
    retrieve_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        "retrieval_complete",
        extra={
            "event": "retrieval_complete",
            "provider": provider_name,
            "embed_model": embed_model,
            "k": k,
            "filter": filter or {},
            "chunk_count": len(docs),
            "retrieve_ms": retrieve_ms,
        },
    )

    t1 = time.perf_counter()
    answer = await generate_answer(question, docs, llm)
    generate_ms = round((time.perf_counter() - t1) * 1000, 2)

    logger.info(
        "generation_complete",
        extra={
            "event": "generation_complete",
            "provider": provider_name,
            "generate_ms": generate_ms,
            "answer_chars": len(answer),
        },
    )

    sources = [
        {
            "source_uri": d.metadata.get("source_uri"),
            "source_type": d.metadata.get("source_type"),
            "chunk_index": d.metadata.get("chunk_index"),
            "tags": d.metadata.get("tags", []),
        }
        for d in docs
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieve_ms": retrieve_ms,
        "generate_ms": generate_ms,
    }
