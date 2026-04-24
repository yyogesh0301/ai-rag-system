import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from api.schemas import IngestResponse
from api.jobs import jobs
from rag.ingest import load_file, SUPPORTED_EXTENSIONS
from rag.chunk import chunk_documents
from rag.db import source_exists
from rag.embed import embed_chunks

DATA_DIR = "data"
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(request: Request, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    job_id = str(uuid.uuid4())
    save_path = str(Path(DATA_DIR) / file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    provider = request.app.state.provider
    vectorstore = request.app.state.vectorstore

    if source_exists(provider.embed_model_name, save_path):
        jobs[job_id] = {"status": "skipped", "filename": file.filename}
        return IngestResponse(job_id=job_id, status="skipped", filename=file.filename)

    jobs[job_id] = {"status": "processing", "filename": file.filename}

    try:
        documents = load_file(save_path)
        chunks = chunk_documents(documents)
        embed_chunks(vectorstore, chunks)
        jobs[job_id]["status"] = "completed"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["detail"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(job_id=job_id, status="completed", filename=file.filename)
