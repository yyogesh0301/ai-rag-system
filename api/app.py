from contextlib import asynccontextmanager
from fastapi import FastAPI
from rag.providers import get_provider
from rag.db import get_vectorstore
from rag.retrieve import get_retriever
from rag.chat import build_chain
from api.routes import ingest, query, status


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider = get_provider()
    embeddings = provider.get_embeddings()
    llm = provider.get_llm()

    vectorstore = get_vectorstore(embeddings, collection_name=provider.embed_model_name)
    retriever = get_retriever(vectorstore, k=5)
    chain = build_chain(retriever, llm)

    app.state.vectorstore = vectorstore
    app.state.provider = provider
    app.state.chain = chain

    yield


app = FastAPI(title="AI RAG System", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(status.router)


@app.get("/health")
def health():
    return {"status": "ok"}
