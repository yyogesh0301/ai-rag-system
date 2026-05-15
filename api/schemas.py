from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    # Operator-style metadata filter passed to PGVector. Examples:
    #   {"source_type": {"$eq": "pdf"}}
    #   {"source_type": {"$in": ["pdf", "code"]}}
    #   {"tags":        {"$in": ["resume"]}}
    filter: dict | None = None


class RetrievedChunk(BaseModel):
    source_uri: str | None = None
    source_type: str | None = None
    chunk_index: int | None = None
    tags: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    retrieve_ms: float | None = None
    generate_ms: float | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str | None = None
    detail: str | None = None
