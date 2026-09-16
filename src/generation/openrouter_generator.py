"""OpenRouter-backed Generator: answers a query from retrieved context via any
chat model OpenRouter exposes, enforcing that every citation marker in the
answer maps to a chunk actually provided.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from src.generation.models import Answer, Citation
from src.generation.openrouter_client import extract_cost_usd, get_openrouter_client
from src.retrieval.models import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")

_SYSTEM_PROMPT = (
    "You are a retrieval-augmented question answering assistant. Answer the "
    "user's question using ONLY the numbered context passages below — never "
    "use outside knowledge. If the passages do not contain the answer, say so "
    "plainly instead of guessing.\n\n"
    "CITATION REQUIREMENT: every factual claim in your answer must end with "
    "an inline citation in square brackets. You MUST copy each id EXACTLY, "
    "character-for-character, from the '[id]' shown immediately before that "
    "passage in the context below — never shorten, renumber, or simplify it. "
    "These ids are long and specific (e.g. '[some-document-p12-c3]'), NOT "
    "short generic ones like '[1]' or '[c1]' — if what you're about to write "
    "looks like the latter, you have the wrong id; go back and copy the real "
    "one. If a claim draws on more than one passage, cite all of them "
    "back-to-back: '...supports this claim [some-document-p12-c3]"
    "[other-document-p4-c1].' Never invent an id, and never cite a passage "
    "that doesn't actually support the claim next to it. A sentence with no "
    "citation is treated as an unverifiable claim, so do not write one — "
    "restructure or omit it instead."
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
            temperature=0,  # deterministic where the provider honors it — see CLAUDE.md
            extra_body={"usage": {"include": True}},  # asks OpenRouter to report cost
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
            model_used=response.model,
            cost_usd=extract_cost_usd(response),
        )

    def generate_stream(self, query: str, context: list[RetrievedChunk]) -> Iterator[str]:
        """Satisfies StreamingGenerator. Yields text deltas as they arrive;
        does not check citations (needs the full text — a marker can land
        anywhere) or extract cost (OpenRouter's cost-on-stream reporting is
        inconsistent enough not to rely on for the live API). Callers that
        need those should accumulate the deltas and call extract_citations()
        themselves once the stream is exhausted (see src/api/app.py)."""
        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            stream=True,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, context)},
            ],
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
