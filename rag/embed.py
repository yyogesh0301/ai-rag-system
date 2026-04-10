from langchain_postgres import PGVector


def embed_chunks(vectorstore: PGVector, chunks: list):
    """Embed chunks and store them in pgvector."""
    vectorstore.add_documents(chunks)
    print(f"Embedded and stored {len(chunks)} chunks.")
