from fastapi import APIRouter, Request
from api.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    chain = request.app.state.chain
    answer = chain.invoke(body.question)
    return QueryResponse(answer=answer)
