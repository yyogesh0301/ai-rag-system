from fastapi import APIRouter, Request

from api.schemas import QueryRequest, QueryResponse
from rag.service import answer_query

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    state = request.app.state
    result = await answer_query(
        question=body.question,
        k=body.k,
        filter=body.filter,
        vectorstore=state.vectorstore,
        llm=state.llm,
        provider_name=state.provider.__class__.__name__,
        embed_model=state.provider.embed_model_name,
    )
    return QueryResponse(**result)
