from rag.providers import get_provider
from rag.ingest import load_file, load_all_files, infer_source_type
from rag.chunk import chunk_documents
from rag.db import get_vectorstore, source_exists
from rag.embed import embed_chunks
from rag.retrieve import get_retriever
from rag.chat import build_chain

DATA_DIR = "data"

provider = get_provider()
embeddings = provider.get_embeddings()
llm = provider.get_llm()

print("Setting up vectorstore...")
vectorstore = get_vectorstore(embeddings, collection_name=provider.embed_model_name)

files = load_all_files(DATA_DIR)
if not files:
    print(f"No supported files found in '{DATA_DIR}/'. Add PDF, CSV, JSON, or TXT files.")
else:
    for file_path in files:
        if source_exists(provider.embed_model_name, file_path):
            print(f"Already indexed: {file_path}")
        else:
            print(f"Ingesting: {file_path}")
            documents = load_file(file_path)
            chunks = chunk_documents(
                documents,
                source_type=infer_source_type(file_path),
                source_uri=file_path,
                embed_model=provider.embed_model_name,
                tags=[],
            )
            print(f"  {len(chunks)} chunks — embedding...")
            embed_chunks(vectorstore, chunks)
            print("  Done.")

retriever = get_retriever(vectorstore, k=5)
chain = build_chain(retriever, llm)

print("\nSystem ready. Ask questions!")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    answer = chain.invoke(question)
    print("AI:", answer)
