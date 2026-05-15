"""Golden Q&A regression suite.

Hits a running instance of the API (default http://localhost:8000) and
asserts retrieval + generation behavior against tests/eval/golden.json.

Run:
    docker compose up -d
    pytest tests/eval -v

Override target:
    RAG_BASE_URL=http://staging:8000 pytest tests/eval -v

This is an integration suite, not a unit test. It exercises the live
provider (Ollama or Gemini), pgvector, and the prompt — exactly what
breaks when retrieval quality regresses.
"""
import json
import os
from pathlib import Path

import httpx
import pytest

BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")
GOLDEN_PATH = Path(__file__).parent / "golden.json"


def _load_cases() -> list[dict]:
    with GOLDEN_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as c:
        # Fail fast if the server isn't up — gives a clearer error than
        # 30 connection-refused tracebacks.
        r = c.get("/health")
        r.raise_for_status()
        yield c


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_golden_case(client: httpx.Client, case: dict):
    payload = {"question": case["question"], "k": case.get("k", 5)}
    if case.get("filter") is not None:
        payload["filter"] = case["filter"]

    r = client.post("/query", json=payload)
    r.raise_for_status()
    body = r.json()

    answer = body["answer"]
    sources = body.get("sources", [])

    expected = case["expected_substring"]
    assert expected.lower() in answer.lower(), (
        f"Answer missing expected substring.\n"
        f"  question: {case['question']}\n"
        f"  expected: {expected!r}\n"
        f"  got:      {answer!r}"
    )

    expected_uri = case.get("expected_source_uri")
    if expected_uri is not None:
        uris = [s.get("source_uri") for s in sources]
        assert expected_uri in uris, (
            f"Expected source_uri {expected_uri!r} not in retrieved sources: {uris}"
        )
