from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents,
    *,
    source_type: str,
    source_uri: str,
    embed_model: str,
    tags: list[str] | None = None,
    extra_metadata: dict | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    """Split documents and stamp each chunk with retrieval-time metadata.

    Metadata written on every chunk:
      - source_type: pdf | csv | json | txt | incident | runbook | markdown
      - source_uri:  absolute or repo-relative path / URI
      - chunk_index: position within the source after splitting
      - embed_model: the embedding model name used for this collection
      - tags:        free-form labels for filtering (e.g. ["resume"])
      - content_hash: sha256 of the raw file (when provided via extra_metadata)
      - (markdown) all YAML frontmatter fields are passed through

    LangChain loaders may already populate metadata (e.g. PDF page numbers).
    We merge — never overwrite — those existing keys; ours win on conflict
    only for the canonical fields above.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    base_meta = {
        **(extra_metadata or {}),
        "source_type": source_type,
        "source_uri": source_uri,
        "embed_model": embed_model,
        "tags": list(tags or []),
    }
    for idx, chunk in enumerate(chunks):
        chunk.metadata = {**chunk.metadata, **base_meta, "chunk_index": idx}
    return chunks
