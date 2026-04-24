import os
import psycopg2
from dotenv import load_dotenv
from langchain_postgres import PGVector

load_dotenv()


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
    """Check if a source is already indexed — direct SQL, no embedding call needed."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""
        SELECT 1
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
        AND e.cmetadata->>'source' = %s
        LIMIT 1;
    """, (collection_name, source))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists
