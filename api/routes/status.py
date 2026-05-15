from fastapi import APIRouter, HTTPException

from api.jobs import get_job
from api.schemas import StatusResponse

router = APIRouter()


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        filename=job.get("filename"),
        detail=job.get("detail"),
    )
