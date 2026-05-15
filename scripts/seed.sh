#!/bin/bash
BASE_URL="http://localhost:8000"

echo "=== Seeding Incidents ==="
for file in data/incidents/*.md; do
  echo "Ingesting: $file"
  curl -s -X POST "$BASE_URL/ingest/file" -F "file=@$file"
  echo ""
done

echo "=== Seeding Runbooks ==="
for file in data/runbooks/*.md; do
  echo "Ingesting: $file"
  curl -s -X POST "$BASE_URL/ingest/file" -F "file=@$file"
  echo ""
done

echo "=== Done ==="
