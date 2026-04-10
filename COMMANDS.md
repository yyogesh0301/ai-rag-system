# Commands Reference

## Docker

### Start DB in background + run app interactively (Q&A mode)
```bash
docker compose up pgvector -d && docker compose run --rm app
```

### Run app interactively (if DB already running)
```bash
docker compose run --rm app
```

### Start everything (logs attached, no interactive input)
```bash
docker compose up --build
```

### Rebuild and restart app only
```bash
docker compose up --build app
```

### Stop all containers
```bash
docker compose down
```

### Stop and remove volumes (wipes DB data)
```bash
docker compose down -v
```

---

## Database

### Connect to pgvector container
```bash
docker exec -it rag_pgvector psql -U postgres -d ragdb
```

### List all tables
```sql
\dt
```

### Drop old document_chunks table (cleanup)
```sql
DROP TABLE document_chunks;
```

### Check indexed sources
```sql
SELECT DISTINCT source FROM chunks_nomic_embed_text;
```

---

## Ollama

### List installed models
```bash
ollama list
```

### Check running models
```bash
ollama ps
```

### Pull embedding model
```bash
ollama pull nomic-embed-text
```

---

## Git

### Push to personal GitHub
```bash
git add .
git commit -m "your message"
git push origin main
```

### Test personal SSH key
```bash
ssh -T git@github-personal
```
