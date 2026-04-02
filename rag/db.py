import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True
    register_vector(conn)
    return conn


def setup_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            source TEXT,
            content TEXT,
            embedding vector(768)
        );
    """)
    cur.close()
    conn.close()


def source_exists(source: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM document_chunks WHERE source = %s LIMIT 1;", (source,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists
