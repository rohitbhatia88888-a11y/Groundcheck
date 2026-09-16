"""Request/response shapes for the live API — deliberately separate from the
internal domain models (src/generation/models.py etc.): the API's public
contract shouldn't change just because an internal model does, and vice
versa.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    """One chunk shown to the generator — the streamed 'done' event lists
    every one of these, not just the cited ones, so the frontend can show
    what was available even when the model didn't cite it."""

    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float
    cited: bool


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
