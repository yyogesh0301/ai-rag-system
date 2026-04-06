from rag.db import get_connection
from rag.providers import get_provider

provider = get_provider()


def embed_chunks(chunks: list[str], source: str):
    conn = get_connection()
    cur = conn.cursor()

    for chunk in chunks:
        embedding = provider.embed(chunk)
        cur.execute(
            "INSERT INTO document_chunks (source, content, embedding, embedding_model) VALUES (%s, %s, %s, %s);",
            (source, chunk, embedding, provider.embed_model_name)
        )
        print(f"Embedded and stored: {chunk[:60]}...")

    cur.close()
    conn.close()
