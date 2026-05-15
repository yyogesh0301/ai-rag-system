from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import json

from api.models.incident import ExtractedMetadata, RCAResponse
from rag.observability import get_logger

logger = get_logger(__name__)


PROMPT_TEMPLATE = """You are a retrieval-grounded assistant.
Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly:
"I don't know based on the provided context."
Do not use outside knowledge. Do not speculate.

Context:
{context}

Question:
{question}"""


def make_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


async def generate_answer(question: str, docs: list, llm) -> str:
    """Generate from already-retrieved docs.

    Used by the API service path so retrieval and generation can be
    timed and logged independently, and so retrieved chunks can be
    returned to the caller.
    """
    chain = make_prompt() | llm | StrOutputParser()
    return await chain.ainvoke({"context": format_docs(docs), "question": question})


CLASSIFY_EXTRACT_PROMPT = """Analyze this text and classify it as "incident" or "runbook". Then extract metadata.

Return ONLY valid JSON, nothing else. Use this exact format:

{{"source_type": "incident", "title": "...", "service": "...", "severity": "critical|high|medium|low", "tags": ["..."], "root_cause_summary": "..."}}
or
{{"source_type": "runbook", "title": "...", "service": "...", "tags": ["..."]}}

Text to analyze:
{content}"""

CLASSIFY_RETRY_PROMPT = CLASSIFY_EXTRACT_PROMPT + """

Your previous response was not valid JSON. The error was:
{parse_error}

Return ONLY valid JSON, nothing else."""


def _parse_extracted_json(raw: str) -> ExtractedMetadata:
    # Strip markdown fences if the LLM wraps in ```json
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(cleaned)
    return ExtractedMetadata(**data)


async def extract_metadata(content: str, llm) -> ExtractedMetadata:
    prompt = ChatPromptTemplate.from_template(CLASSIFY_EXTRACT_PROMPT)
    chain = prompt | llm | StrOutputParser()
    raw = await chain.ainvoke({"content": content[:3000]})

    try:
        return _parse_extracted_json(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("extract_metadata_retry", extra={"event": "extract_metadata_retry", "error": str(e)})

    retry_prompt = ChatPromptTemplate.from_template(CLASSIFY_RETRY_PROMPT)
    retry_chain = retry_prompt | llm | StrOutputParser()
    raw = await retry_chain.ainvoke({"content": content[:3000], "parse_error": str(e)})

    try:
        return _parse_extracted_json(raw)
    except (json.JSONDecodeError, Exception) as e2:
        logger.error("extract_metadata_failed", extra={"event": "extract_metadata_failed", "error": str(e2)})
        return ExtractedMetadata(source_type="incident", title="Untitled")


RCA_SYSTEM_PROMPT = """You are an incident response assistant for a DevOps/SRE team. Given a new alert or error log, and retrieved context from past incidents and runbooks, generate a structured root cause analysis.

The retrieval system has already computed a confidence level from similarity scores. This is a FACT provided to you. Do not override it.

Rules:
- Use ONLY information present in the retrieved context
- Cite specific incident IDs when referencing past cases
- Cite specific runbook IDs and sections when suggesting fixes
- NEVER fabricate details not in the context
- If NO similar incidents were found: say "No matching past incident found" and suggest troubleshooting from runbooks if available
- If runbooks are also empty: suggest general debugging steps
- Output must be valid JSON matching the schema

Retrieval confidence (COMPUTED, do not override): {confidence_level} (top-1 similarity: {top1_score}, score spread: {topk_spread})

Context from similar past incidents:
{incident_context}

Context from relevant runbooks:
{runbook_context}

Current alert/error:
{query}

{format_instructions}"""


def _format_incident_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(none)"
    parts = []
    for c in chunks:
        meta = c["metadata"]
        incident_id = meta.get("incident_id", "unknown")
        title = meta.get("title", "untitled")
        service = meta.get("service", "unknown")
        parts.append(
            f"[{incident_id}] {title} (service: {service})\n{c['page_content']}"
        )
    return "\n\n---\n\n".join(parts)


def _format_runbook_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(none)"
    parts = []
    for c in chunks:
        meta = c["metadata"]
        runbook_id = meta.get("runbook_id", "unknown")
        title = meta.get("title", "untitled")
        parts.append(f"[{runbook_id}] {title}\n{c['page_content']}")
    return "\n\n---\n\n".join(parts)


async def generate_rca(
    query: str,
    retrieval_results: dict[str, list[dict]],
    confidence: dict,
    llm,
) -> RCAResponse:
    parser = PydanticOutputParser(pydantic_object=RCAResponse)
    prompt = ChatPromptTemplate.from_template(RCA_SYSTEM_PROMPT)

    incident_context = _format_incident_context(
        retrieval_results.get("incident", [])
    )
    runbook_context = _format_runbook_context(
        retrieval_results.get("runbook", [])
    )

    chain = prompt | llm | parser
    prompt_vars = {
        "confidence_level": confidence["level"],
        "top1_score": confidence["top1_score"],
        "topk_spread": confidence["topk_spread"],
        "incident_context": incident_context,
        "runbook_context": runbook_context,
        "query": query,
        "format_instructions": parser.get_format_instructions(),
    }

    try:
        return await chain.ainvoke(prompt_vars)
    except OutputParserException as e:
        logger.warning(
            "rca_parse_retry",
            extra={"event": "rca_parse_retry", "error": str(e)},
        )
        # Retry once: append the parse error so the LLM can self-correct
        retry_prompt = ChatPromptTemplate.from_template(
            RCA_SYSTEM_PROMPT
            + "\n\nYour previous response failed to parse with error:\n{parse_error}\n\nFix the JSON and try again."
        )
        retry_chain = retry_prompt | llm | parser
        try:
            return await retry_chain.ainvoke({**prompt_vars, "parse_error": str(e)})
        except OutputParserException as e2:
            logger.error(
                "rca_parse_failed",
                extra={"event": "rca_parse_failed", "error": str(e2)},
            )
            return RCAResponse(
                probable_root_cause="Unable to parse structured response from LLM",
                confidence_level=confidence["level"],
                summary=f"RCA generation failed after retry. Raw query: {query[:200]}",
            )


def build_chain(retriever, llm):
    """Backwards-compatible LCEL chain for the CLI entrypoint (main.py).

    The API path uses `generate_answer` directly so it can return sources
    and emit per-stage logs.
    """
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | make_prompt()
        | llm
        | StrOutputParser()
    )
