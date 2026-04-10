from langchain_postgres import PGVector


def get_retriever(vectorstore: PGVector, k: int = 5):
    """Return a LangChain retriever from the vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
