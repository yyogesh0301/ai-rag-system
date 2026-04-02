from google import genai
import os
from dotenv import load_dotenv
from rag.db import get_connection

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def retrieve(query: str, k: int = 5) -> list[str]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query]
    )
    query_embedding = response.embeddings[0].values

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT content
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [row[0] for row in rows]
