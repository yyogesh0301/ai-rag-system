from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    input_text: str = Field(description="Alert message, error log, or symptom description")
    top_k: int = Field(default=5, ge=1, le=20)


class ExtractedMetadata(BaseModel):
    source_type: str = Field(description="incident or runbook")
    title: str
    service: str = ""
    severity: str = ""
    tags: list[str] = Field(default_factory=list)
    root_cause_summary: str = ""


class IngestJobResult(BaseModel):
    job_id: str
    source_uri: str
    source_type: str
    extracted_metadata: ExtractedMetadata | None = None


class UnifiedIngestResponse(BaseModel):
    jobs: list[IngestJobResult]


class SimilarIncident(BaseModel):
    incident_id: str
    title: str
    similarity: str = Field(description="Why this past incident is relevant")


class SuggestedAction(BaseModel):
    action: str
    source: str = Field(description="Incident ID or Runbook ID:section that justifies this")


class RCAResponse(BaseModel):
    probable_root_cause: str = Field(description="Most likely root cause based on retrieved context")
    confidence_level: str = Field(description="high, medium, or low — as provided by retrieval system")
    similar_incidents: list[SimilarIncident] = Field(default_factory=list)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    runbook_refs: list[str] = Field(default_factory=list, description="Runbook IDs referenced")
    summary: str = Field(description="1-2 sentence executive summary for the on-call engineer")
