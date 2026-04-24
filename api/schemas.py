from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str


class IngestResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str | None = None
    detail: str | None = None
