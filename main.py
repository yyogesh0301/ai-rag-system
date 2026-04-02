from rag.db import setup_db, source_exists
from rag.ingest import load_pdf
from rag.chunk import chunk_text
from rag.embed import embed_chunks
from rag.chat import ask

PDF_PATH = "data/yy.pdf"

print("Setting up DB...")
setup_db()

if source_exists(PDF_PATH):
    print(f"'{PDF_PATH}' already indexed. Skipping ingestion.")
else:
    print("Loading PDF...")
    text = load_pdf(PDF_PATH)

    print("Chunking...")
    chunks = chunk_text(text)

    print(f"Embedding and storing {len(chunks)} chunks...")
    embed_chunks(chunks, source=PDF_PATH)

print("System ready. Ask questions!")

while True:
    question = input("You: ")
    answer = ask(question)
    print("AI:", answer)
