from rag.db import get_connection
from rag.providers import get_provider

provider = get_provider()


def retrieve(query: str, k: int = 5) -> list[str]:
    query_embedding = provider.embed(query)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT content
        FROM document_chunks
        WHERE embedding_model = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (provider.embed_model_name, query_embedding, k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [row[0] for row in rows]
