import os

import psycopg2
from dotenv import load_dotenv
from langchain_postgres import PGVector

from rag.observability import get_logger

load_dotenv()

logger = get_logger(__name__)


def get_vectorstore(embeddings, collection_name: str) -> PGVector:
    """Return a PGVector store for the given embeddings and collection.

    PGVector automatically:
    - Creates the table if it doesn't exist
    - Detects embedding dimensions from the model
    - Isolates collections by name (one per embedding model)
    """
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=os.getenv("DATABASE_URL"),
    )


def source_exists(collection_name: str, source: str) -> bool:
    """Check if a source is already indexed.

    Looks at both `source_uri` (what we now write) and the legacy `source`
    key that older chunks may carry, so re-ingest detection keeps working
    across the metadata schema change.
    """
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
          AND (e.cmetadata->>'source_uri' = %s OR e.cmetadata->>'source' = %s)
        LIMIT 1;
        """,
        (collection_name, source, source),
    )
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists


def get_source_content_hash(collection_name: str, source: str) -> str | None:
    """Return the content_hash stored on existing chunks for this source, or None."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.cmetadata->>'content_hash'
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
          AND (e.cmetadata->>'source_uri' = %s OR e.cmetadata->>'source' = %s)
        LIMIT 1;
        """,
        (collection_name, source, source),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None


def delete_source_chunks(collection_name: str, source: str) -> int:
    """Delete all chunks for a given source_uri. Returns count of deleted rows."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = %s
        )
        AND (cmetadata->>'source_uri' = %s OR cmetadata->>'source' = %s);
        """,
        (collection_name, source, source),
    )
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    logger.info(
        "source_chunks_deleted",
        extra={"event": "source_chunks_deleted", "source_uri": source, "deleted": deleted},
    )
    return deleted


def ensure_indexes() -> None:
    """Create the HNSW index on the embedding column if it does not exist.

    pgvector needs at least one row to infer the vector dimensions for
    HNSW.  If the table is empty we skip and log — the index will be
    created on the next startup after data has been ingested.
    """
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM langchain_pg_embedding LIMIT 1);"
        )
        has_rows = cur.fetchone()[0]

        if not has_rows:
            cur.close()
            conn.close()
            logger.info(
                "hnsw_index_deferred",
                extra={"event": "hnsw_index_deferred", "reason": "table empty, index deferred until data exists"},
            )
            return

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_cosine
            ON langchain_pg_embedding
            USING hnsw (embedding vector_cosine_ops);
            """
        )
        cur.close()
        conn.close()
        logger.info("hnsw_index_ready", extra={"event": "hnsw_index_ready"})
    except Exception as e:
        logger.warning(
            "hnsw_index_skipped",
            extra={"event": "hnsw_index_skipped", "reason": str(e)},
        )


def ensure_jobs_table() -> None:
    """Create the jobs table for ingest job tracking. Idempotent."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id          UUID PRIMARY KEY,
            status      TEXT NOT NULL,
            filename    TEXT,
            source_uri  TEXT,
            detail      TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    cur.close()
    conn.close()
    logger.info("jobs_table_ready", extra={"event": "jobs_table_ready"})
