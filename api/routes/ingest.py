import hashlib
import shutil
import time
import uuid
from pathlib import Path
import frontmatter
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from langchain_core.documents import Document

from api.jobs import create_job, update_job
from api.models.incident import ExtractedMetadata, IngestJobResult, UnifiedIngestResponse
from rag.chat import extract_metadata
from rag.chunk import chunk_documents
from rag.db import delete_source_chunks, get_source_content_hash, source_exists
from rag.embed import embed_chunks
from rag.ingest import (
    SUPPORTED_EXTENSIONS,
    _sanitize_metadata,
    _split_by_headings,
    infer_source_type,
    load_file,
)
from rag.observability import get_logger

DATA_DIR = "data"
router = APIRouter()
logger = get_logger(__name__)


def _content_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dedupe_check(
    collection_name: str, source_uri: str, content_hash: str, force: bool
) -> str | None:
    """Return 'skip' if unchanged, None to proceed. Deletes old chunks if hash changed or force."""
    if force:
        if source_exists(collection_name, source_uri):
            delete_source_chunks(collection_name, source_uri)
        return None

    existing_hash = get_source_content_hash(collection_name, source_uri)
    if existing_hash is None:
        return None
    if existing_hash == content_hash:
        return "skip"
    delete_source_chunks(collection_name, source_uri)
    return None


def _run_ingest_job(
    *,
    job_id: str,
    source_uri: str,
    source_type: str,
    documents: list[Document],
    tags: list[str],
    embed_model: str,
    content_hash: str,
    vectorstore,
) -> None:
    try:
        chunks = chunk_documents(
            documents,
            source_type=source_type,
            source_uri=source_uri,
            embed_model=embed_model,
            tags=tags,
            extra_metadata={"content_hash": content_hash},
        )
        embed_chunks(vectorstore, chunks)
        update_job(job_id=job_id, status="completed")
        logger.info(
            "ingest_completed",
            extra={
                "event": "ingest_completed",
                "job_id": job_id,
                "source_uri": source_uri,
                "source_type": source_type,
                "chunk_count": len(chunks),
            },
        )
    except Exception as e:
        update_job(job_id=job_id, status="failed", detail=str(e))
        logger.exception(
            "ingest_failed",
            extra={"event": "ingest_failed", "job_id": job_id, "source_uri": source_uri},
        )


def _md_has_frontmatter(path: str) -> tuple[bool, dict, str]:
    """Parse a markdown file. Returns (has_frontmatter, metadata_dict, body)."""
    post = frontmatter.load(path)
    meta = _sanitize_metadata(dict(post.metadata)) if post.metadata else {}
    return bool(post.metadata), meta, post.content


def _build_documents(body: str, metadata: dict) -> list[Document]:
    sections = _split_by_headings(body)
    if not sections:
        sections = [body]
    return [
        Document(page_content=s, metadata={**metadata, "section_index": i})
        for i, s in enumerate(sections)
    ]


def _generate_id(source_type: str) -> str:
    ts = int(time.time())
    return f"INC-{ts}" if source_type == "incident" else f"RB-{ts}"


async def _process_file(
    file: UploadFile,
    *,
    force: bool,
    llm,
    collection_name: str,
    embed_model: str,
    vectorstore,
    background_tasks: BackgroundTasks,
) -> IngestJobResult:
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}",
        )

    job_id = str(uuid.uuid4())
    save_path = Path(DATA_DIR) / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    source_uri = save_path.as_posix()

    raw_bytes = await file.read()
    with open(save_path, "wb") as f:
        f.write(raw_bytes)

    content_hash = _content_hash_bytes(raw_bytes)
    extracted: ExtractedMetadata | None = None

    if _dedupe_check(collection_name, source_uri, content_hash, force) == "skip":
        create_job(job_id=job_id, status="skipped", filename=file.filename, source_uri=source_uri)
        return IngestJobResult(job_id=job_id, source_uri=source_uri, source_type="skipped")

    if ext == ".md":
        has_fm, meta, body = _md_has_frontmatter(str(save_path))
        if has_fm:
            source_type = meta.get("source_type", infer_source_type(source_uri))
            # Frontmatter may not have source_type — infer from incident_id/runbook_id keys
            if "incident_id" in meta:
                source_type = "incident"
            elif "runbook_id" in meta:
                source_type = "runbook"
            documents = _build_documents(body, meta)
            tags = meta.get("tags", [])
        else:
            extracted = await extract_metadata(body, llm)
            source_type = extracted.source_type
            meta = extracted.model_dump()
            meta[
                "incident_id" if source_type == "incident" else "runbook_id"
            ] = _generate_id(source_type)
            documents = _build_documents(body, meta)
            tags = extracted.tags
    else:
        documents = load_file(source_uri)
        source_type = infer_source_type(source_uri)
        # For non-markdown with text content, try LLM extraction
        if ext == ".txt":
            text = raw_bytes.decode("utf-8", errors="replace")
            extracted = await extract_metadata(text, llm)
            source_type = extracted.source_type
            meta = extracted.model_dump()
            meta[
                "incident_id" if source_type == "incident" else "runbook_id"
            ] = _generate_id(source_type)
            for doc in documents:
                doc.metadata.update(meta)
            tags = extracted.tags
        else:
            tags = []

    create_job(job_id=job_id, status="processing", filename=file.filename, source_uri=source_uri)

    background_tasks.add_task(
        _run_ingest_job,
        job_id=job_id,
        source_uri=source_uri,
        source_type=source_type,
        documents=documents,
        tags=tags if isinstance(tags, list) else [],
        embed_model=embed_model,
        content_hash=content_hash,
        vectorstore=vectorstore,
    )

    return IngestJobResult(
        job_id=job_id,
        source_uri=source_uri,
        source_type=source_type,
        extracted_metadata=extracted,
    )


async def _process_text(
    content: str,
    *,
    force: bool,
    llm,
    collection_name: str,
    embed_model: str,
    vectorstore,
    background_tasks: BackgroundTasks,
) -> IngestJobResult:
    job_id = str(uuid.uuid4())
    source_uri = f"text://{job_id}"
    content_hash = _content_hash_bytes(content.encode())

    extracted = await extract_metadata(content, llm)
    source_type = extracted.source_type
    generated_id = _generate_id(source_type)

    meta = extracted.model_dump()
    meta["incident_id" if source_type == "incident" else "runbook_id"] = generated_id

    if _dedupe_check(collection_name, source_uri, content_hash, force) == "skip":
        create_job(job_id=job_id, status="skipped", filename=generated_id, source_uri=source_uri)
        return IngestJobResult(
            job_id=job_id, source_uri=source_uri, source_type=source_type, extracted_metadata=extracted
        )

    documents = _build_documents(content, meta)

    create_job(job_id=job_id, status="processing", filename=generated_id, source_uri=source_uri)

    background_tasks.add_task(
        _run_ingest_job,
        job_id=job_id,
        source_uri=source_uri,
        source_type=source_type,
        documents=documents,
        tags=extracted.tags,
        embed_model=embed_model,
        content_hash=content_hash,
        vectorstore=vectorstore,
    )

    return IngestJobResult(
        job_id=job_id,
        source_uri=source_uri,
        source_type=source_type,
        extracted_metadata=extracted,
    )


@router.post("/ingest/file", response_model=IngestJobResult, tags=["ingest"])
async def ingest_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    state = request.app.state
    collection_name = state.provider.embed_model_name

    return await _process_file(
        file,
        force=False,
        llm=state.llm,
        collection_name=collection_name,
        embed_model=collection_name,
        vectorstore=state.vectorstore,
        background_tasks=background_tasks,
    )


@router.post("/ingest/text", response_model=IngestJobResult, tags=["ingest"])
async def ingest_text(
    request: Request,
    background_tasks: BackgroundTasks,
    content: str = Form(...),
):
    state = request.app.state
    collection_name = state.provider.embed_model_name

    result = await _process_text(
        content,
        force=True,
        llm=state.llm,
        collection_name=collection_name,
        embed_model=collection_name,
        vectorstore=state.vectorstore,
        background_tasks=background_tasks,
    )

    return result
