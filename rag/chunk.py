from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    """Split LangChain Document objects into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)
