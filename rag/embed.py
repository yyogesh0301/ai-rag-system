from google import genai
import os
from dotenv import load_dotenv
from rag.db import get_connection

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def embed_chunks(chunks: list[str], source: str):
    conn = get_connection()
    cur = conn.cursor()

    for chunk in chunks:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[chunk]
        )
        embedding = response.embeddings[0].values
        cur.execute(
            "INSERT INTO document_chunks (source, content, embedding) VALUES (%s, %s, %s);",
            (source, chunk, embedding)
        )
        print(f"Embedded and stored: {chunk[:60]}...")

    cur.close()
    conn.close()
