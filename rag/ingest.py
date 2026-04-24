from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)
from langchain_community.document_loaders import JSONLoader


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".json", ".txt"}


def load_file(path: str):
    """Load a file and return a list of LangChain Document objects.
    Supports: PDF, CSV, JSON, TXT
    """
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(path).load()

    elif ext == ".csv":
        return CSVLoader(path).load()

    elif ext == ".json":
        return JSONLoader(
            file_path=path,
            jq_schema=".",
            text_content=False
        ).load()

    elif ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()

    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")


def load_all_files(data_dir: str) -> list:
    """Scan a directory and return (file_path, documents) for each supported file."""
    files = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(path))
    return files
