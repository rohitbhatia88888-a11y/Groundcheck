"""OpenRouter-backed Generator: answers a query from retrieved context via any
chat model OpenRouter exposes, enforcing that every citation marker in the
answer maps to a chunk actually provided.
"""

from __future__ import annotations

import re

from src.generation.models import Answer, Citation
from src.generation.openrouter_client import get_openrouter_client
from src.retrieval.models import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")

_SYSTEM_PROMPT = (
    "You are a retrieval-augmented question answering assistant. Answer the "
    "user's question using ONLY the numbered context passages below — never "
    "use outside knowledge. If the passages do not contain the answer, say so "
    "plainly instead of guessing.\n\n"
    "Cite every claim inline with the passage's id in square brackets, e.g. "
    "'The sky is blue [c1].' Use exactly the ids given, and cite every "
    "passage you actually rely on."
)


def _build_user_message(query: str, context: list[RetrievedChunk]) -> str:
    context_block = "\n\n".join(
        f"[{chunk.chunk_id}] (doc={chunk.metadata.doc_id}, page={chunk.metadata.page})\n{chunk.text}"
        for chunk in context
    )
    return f"Context passages:\n\n{context_block}\n\nQuestion: {query}"


def extract_citations(
    text: str, context: list[RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    """Parses bracketed markers out of generated text and enforces that each one
    maps to a chunk actually present in `context`.

    Returns (valid citations, de-duplicated in first-seen order;
             marker strings that did NOT match any chunk_id in context).
    """
    chunks_by_id = {chunk.chunk_id: chunk for chunk in context}

    citations: list[Citation] = []
    unsupported: list[str] = []
    seen: set[str] = set()

    for marker in _CITATION_PATTERN.findall(text):
        chunk = chunks_by_id.get(marker)
        if chunk is None:
            unsupported.append(marker)
            continue
        if marker in seen:
            continue
        seen.add(marker)
        citations.append(
            Citation(chunk_id=marker, doc_id=chunk.metadata.doc_id, page=chunk.metadata.page)
        )

    return citations, unsupported


class OpenRouterGenerator:
    """Generator backed by any chat model exposed through OpenRouter."""

    def __init__(self, model: str = "openai/gpt-4o-mini", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = get_openrouter_client()

    def generate(self, query: str, context: list[RetrievedChunk]) -> Answer:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, context)},
            ],
        )
        text = response.choices[0].message.content or ""
        citations, unsupported = extract_citations(text, context)

        return Answer(
            text=text,
            citations=citations,
            context=context,
            unsupported_citation_markers=unsupported,
        )
