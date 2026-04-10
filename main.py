from rag.providers import get_provider
from rag.ingest import load_pdf
from rag.chunk import chunk_documents
from rag.db import get_vectorstore, source_exists
from rag.embed import embed_chunks
from rag.retrieve import get_retriever
from rag.chat import build_chain

PDF_PATH = "data/yy.pdf"

provider = get_provider()
embeddings = provider.get_embeddings()
llm = provider.get_llm()

print("Setting up vectorstore...")
vectorstore = get_vectorstore(embeddings, collection_name=provider.embed_model_name)

if source_exists(vectorstore, PDF_PATH):
    print(f"'{PDF_PATH}' already indexed. Skipping ingestion.")
else:
    print("Loading PDF...")
    documents = load_pdf(PDF_PATH)

    print("Chunking...")
    chunks = chunk_documents(documents)

    print(f"Embedding and storing {len(chunks)} chunks...")
    embed_chunks(vectorstore, chunks)

retriever = get_retriever(vectorstore, k=5)
chain = build_chain(retriever, llm)

print("System ready. Ask questions!")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    answer = chain.invoke(question)
    print("AI:", answer)
