import re
from datetime import date, datetime
from pathlib import Path

import frontmatter
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".json", ".txt", ".md"}

_EXT_TO_SOURCE_TYPE = {
    ".pdf": "pdf",
    ".csv": "csv",
    ".json": "json",
    ".txt": "txt",
    ".md": "markdown",
}

# Directory-based overrides for markdown files.
_DIR_TO_SOURCE_TYPE = {
    "incidents": "incident",
    "runbooks": "runbook",
}


def infer_source_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in _EXT_TO_SOURCE_TYPE:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")
    if ext == ".md":
        parent = Path(path).parent.name.lower()
        return _DIR_TO_SOURCE_TYPE.get(parent, "markdown")
    return _EXT_TO_SOURCE_TYPE[ext]


def _split_by_headings(body: str) -> list[str]:
    """Split markdown body into sections by ## headings.

    Each returned string starts with the heading line.
    Content before the first ## heading (if any) becomes its own chunk.
    """
    parts = re.split(r"(?=^## )", body, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _sanitize_value(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, list):
        return [_sanitize_value(item) for item in v]
    if isinstance(v, dict):
        return _sanitize_metadata(v)
    return v


def _sanitize_metadata(metadata: dict) -> dict:
    return {k: _sanitize_value(v) for k, v in metadata.items()}


def load_markdown(path: str) -> list[Document]:
    post = frontmatter.load(path)
    meta = _sanitize_metadata(dict(post.metadata))
    sections = _split_by_headings(post.content)
    if not sections:
        sections = [post.content]
    return [
        Document(page_content=section, metadata={**meta, "section_index": i})
        for i, section in enumerate(sections)
    ]


def load_file(path: str):
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(path).load()

    elif ext == ".csv":
        return CSVLoader(path).load()

    elif ext == ".json":
        return JSONLoader(
            file_path=path,
            jq_schema=".",
            text_content=False,
        ).load()

    elif ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()

    elif ext == ".md":
        return load_markdown(path)

    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")


def load_all_files(data_dir: str) -> list:
    files = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(path))
    return files
