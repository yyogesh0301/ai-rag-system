# AI RAG System

A backend-first knowledge retrieval platform that lets you query documents using natural language. Built with a modular provider architecture that supports both local models (Ollama) and cloud APIs (Gemini) — switchable via a single environment variable.

## Features

- **Multi-provider support** — Ollama (local) or Gemini (API), switch with `LLM_PROVIDER` in `.env`
- **pgvector storage** — embeddings persisted in PostgreSQL with vector similarity search
- **Model isolation** — separate vector collections per embedding model, no cross-model comparison issues
- **Smart chunking** — `RecursiveCharacterTextSplitter` for better context boundaries
- **Context-aware answers** — uses retrieved context when relevant, falls back to model knowledge otherwise
- **Multi-source ingestion** — drop PDF, CSV, JSON, or TXT files into `data/`, auto-ingested on startup
- **REST API** — FastAPI endpoints for `/ingest`, `/query`, `/status/{job_id}`
- **Async ingestion** — background workers with Celery, job state tracking *(in progress)*

## Tech Stack

- **Framework**: FastAPI + Uvicorn (Python)
- **Vector DB**: PostgreSQL + pgvector
- **LLM / Embeddings**: Ollama (`gemma3:4b`, `nomic-embed-text`) / Google Gemini
- **Orchestration**: LangChain (LCEL chains, PGVector, document loaders) + LangGraph
- **Deployment**: Docker Compose

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Ollama running locally with `nomic-embed-text` and `gemma3:4b` pulled

```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

### Setup

```bash
git clone https://github.com/yyogesh0301/ai-rag-system.git
cd ai-rag-system
cp .env.example .env   # fill in your values
```

### Run

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload a file (PDF, CSV, JSON, TXT) for ingestion |
| `POST` | `/query` | Ask a question, get a context-aware answer |
| `GET` | `/status/{job_id}` | Check ingestion job status |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

### Examples

```bash
# Upload a file
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/yy.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is my name?"}'

# Check job status
curl http://localhost:8000/status/<job_id>
```

## Ingestion

Drop any supported file into the `data/` folder — the app auto-ingests on startup and skips already-indexed files.

```
data/
  resume.pdf
  employees.csv
  config.json
  notes.txt
```

Supported formats: `.pdf`, `.csv`, `.json`, `.txt`

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `ollama` or `gemini` | `ollama` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL` | Embedding model | `nomic-embed-text` |
| `OLLAMA_GENERATE_MODEL` | Generation model | `gemma3:4b` |
| `GEMINI_API_KEY` | Gemini API key | — |
| `DATABASE_URL` | PostgreSQL connection string | — |

## Project Structure

```
ai-rag-system/
├── api/
│   ├── routes/
│   │   ├── ingest.py     # POST /ingest
│   │   ├── query.py      # POST /query
│   │   └── status.py     # GET /status/{job_id}
│   ├── app.py            # FastAPI app + lifespan
│   ├── jobs.py           # Job state tracking
│   └── schemas.py        # Pydantic models
├── rag/
│   ├── providers/        # LLM provider abstraction (Ollama / Gemini)
│   ├── ingest.py         # Document loading (PDF, CSV, JSON, TXT)
│   ├── chunk.py          # Text splitting
│   ├── db.py             # pgvector setup
│   ├── embed.py          # Embedding + storage
│   ├── retrieve.py       # Similarity search
│   └── chat.py           # LCEL chain
├── data/                 # Drop documents here
├── server.py             # Uvicorn entry point
└── docker-compose.yml
```

## LangChain Architecture

| Component | LangChain Class | Role |
|-----------|----------------|------|
| Document loading | `PyPDFLoader`, `CSVLoader`, `JSONLoader`, `TextLoader` | Loads files with metadata |
| Chunking | `RecursiveCharacterTextSplitter` | Splits on paragraphs → sentences → words |
| Vector store | `PGVector` | Auto-detects embedding dimensions, manages collections |
| Retrieval | `vectorstore.as_retriever()` | Cosine similarity search, top-k |
| Chain | LCEL (`\|` operator) | `retriever → prompt → llm → parser` |
| Routing | LangGraph | Conditional context relevance check *(in progress)* |

## Switching Providers

```env
# Use local Ollama (no API key needed)
LLM_PROVIDER=ollama

# Use Google Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

No code changes required — the provider pattern handles the rest.
