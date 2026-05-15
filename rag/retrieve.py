from langchain_postgres import PGVector


def get_retriever(
    vectorstore: PGVector,
    k: int = 5,
    filter: dict | None = None,
):
    """Build a retriever per request.

    `filter` is passed straight to PGVector. langchain-postgres expects
    operator-style dicts on JSONB metadata, e.g.:
        {"source_type": {"$eq": "pdf"}}
        {"source_type": {"$in": ["pdf", "code"]}}
        {"tags":        {"$in": ["resume"]}}

    None or {} disables filtering — important so we don't pass an empty
    dict to the SQL builder, which some versions interpret as a no-op
    and others reject.
    """
    search_kwargs: dict = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def _search_one_source(
    vectorstore: PGVector,
    query: str,
    source_type: str,
    k: int,
) -> list[dict]:
    """Run sync similarity_search_with_score for a single source_type filter."""
    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter={"source_type": {"$eq": source_type}},
    )
    # PGVector returns (Document, distance). Convert distance → similarity.
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(1.0 - distance, 4),
        }
        for doc, distance in results
    ]


async def retrieve_multi_source(
    vectorstore: PGVector,
    query: str,
    k_per_source: int = 3,
    sources: list[str] | None = None,
) -> dict[str, list[dict]]:
    if sources is None:
        sources = ["incident", "runbook"]

    return {
        src: _search_one_source(vectorstore, query, src, k_per_source)
        for src in sources
    }


def compute_confidence(retrieval_results: dict[str, list[dict]]) -> dict:
    all_scores = [
        chunk["similarity_score"]
        for chunks in retrieval_results.values()
        for chunk in chunks
    ]
    if not all_scores:
        return {"level": "low", "top1_score": 0.0, "topk_spread": 0.0}

    all_scores.sort(reverse=True)
    top1 = all_scores[0]
    topk_spread = round(top1 - all_scores[-1], 4)

    if top1 > 0.85:
        level = "high"
    elif top1 > 0.65:
        level = "medium"
    else:
        level = "low"

    return {"level": level, "top1_score": round(top1, 4), "topk_spread": topk_spread}
