import os
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


def source_exists(vectorstore: PGVector, source: str) -> bool:
    """Check if a source has already been indexed in this collection."""
    results = vectorstore.similarity_search(
        query="",
        k=1,
        filter={"source": source}
    )
    return len(results) > 0
