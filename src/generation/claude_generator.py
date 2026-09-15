"""Claude-backed Generator: answers a query from retrieved context, enforcing
that every citation marker in the answer maps to a chunk actually provided.

Requires ANTHROPIC_API_KEY in the environment (read automatically by the
Anthropic client) — never pass a key as a literal string.
"""

from __future__ import annotations

import re

import anthropic

from src.generation.models import Answer, Citation
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

    Returns (valid citations, in first-seen order and de-duplicated;
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


class ClaudeGenerator:
    """Generator backed by the Anthropic Messages API."""

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def generate(self, query: str, context: list[RetrievedChunk]) -> Answer:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(query, context)}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        citations, unsupported = extract_citations(text, context)

        return Answer(
            text=text,
            citations=citations,
            context=context,
            unsupported_citation_markers=unsupported,
        )
