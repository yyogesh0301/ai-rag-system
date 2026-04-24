from fastapi import APIRouter, HTTPException
from api.schemas import StatusResponse
from api.jobs import jobs

router = APIRouter()


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        filename=job.get("filename"),
        detail=job.get("detail"),
    )
