from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str):
    """Load a PDF and return a list of LangChain Document objects (one per page)."""
    loader = PyPDFLoader(path)
    return loader.load()
