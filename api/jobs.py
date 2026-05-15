"""Postgres-backed job state for ingest tracking.

Kept as plain psycopg2 calls to stay consistent with rag/db.py and avoid
pulling in an ORM. If/when sessions, evals, or other tables join, this is
the right time to introduce Tortoise — not before.
"""
import os

import psycopg2
import psycopg2.extras


def _conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def create_job(
    *,
    job_id: str,
    status: str,
    filename: str | None = None,
    source_uri: str | None = None,
) -> None:
    conn = _conn()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs (id, status, filename, source_uri)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET status = EXCLUDED.status,
                filename = EXCLUDED.filename,
                source_uri = EXCLUDED.source_uri,
                updated_at = now();
        """,
        (job_id, status, filename, source_uri),
    )
    cur.close()
    conn.close()


def update_job(*, job_id: str, status: str, detail: str | None = None) -> None:
    conn = _conn()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE jobs
           SET status = %s,
               detail = COALESCE(%s, detail),
               updated_at = now()
         WHERE id = %s;
        """,
        (status, detail, job_id),
    )
    cur.close()
    conn.close()


def get_job(job_id: str) -> dict | None:
    conn = _conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, status, filename, source_uri, detail FROM jobs WHERE id = %s;",
        (job_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None
