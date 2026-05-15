import time

from fastapi import APIRouter, HTTPException, Request

from api.models.incident import DiagnoseRequest, RCAResponse
from rag.chat import generate_rca
from rag.observability import get_logger
from rag.retrieve import compute_confidence, retrieve_multi_source

router = APIRouter(prefix="/diagnose", tags=["diagnose"])
logger = get_logger(__name__)


@router.post("/", response_model=RCAResponse)
async def diagnose(request: Request, body: DiagnoseRequest):
    state = request.app.state

    try:
        t0 = time.perf_counter()
        retrieval_results = await retrieve_multi_source(
            vectorstore=state.vectorstore,
            query=body.input_text,
            k_per_source=body.top_k,
            sources=["incident", "runbook"],
        )
        retrieve_ms = round((time.perf_counter() - t0) * 1000, 2)

        confidence = compute_confidence(retrieval_results)

        logger.info(
            "diagnose_retrieval_complete",
            extra={
                "event": "diagnose_retrieval_complete",
                "retrieve_ms": retrieve_ms,
                "confidence": confidence,
                "incident_chunks": len(retrieval_results.get("incident", [])),
                "runbook_chunks": len(retrieval_results.get("runbook", [])),
            },
        )

        t1 = time.perf_counter()
        rca = await generate_rca(
            query=body.input_text,
            retrieval_results=retrieval_results,
            confidence=confidence,
            llm=state.llm,
        )
        generate_ms = round((time.perf_counter() - t1) * 1000, 2)

        logger.info(
            "diagnose_complete",
            extra={
                "event": "diagnose_complete",
                "retrieve_ms": retrieve_ms,
                "generate_ms": generate_ms,
                "confidence_level": confidence["level"],
            },
        )

        return rca

    except Exception as e:
        logger.exception(
            "diagnose_failed",
            extra={"event": "diagnose_failed", "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))
