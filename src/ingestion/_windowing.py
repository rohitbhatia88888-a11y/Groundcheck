"""Shared sliding-window helper for the fixed-size chunkers (FixedSizeChunker
over characters, FixedTokenChunker over tokens) — same windowing logic, just
applied to a different unit."""

from __future__ import annotations

from collections.abc import Iterator


def validate_window_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )


def sliding_windows(length: int, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Yields (start, end) index pairs covering a sequence of `length` items in
    overlapping windows of `size`, advancing by `size - overlap` each step."""
    stride = size - overlap
    start = 0
    while start < length:
        yield start, start + size
        start += stride
