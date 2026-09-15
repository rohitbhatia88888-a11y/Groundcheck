"""Generates a candidate (question, answer) pair from a single chunk, via
OpenRouter — one LLM call per sampled chunk, forced tool-call output so the
result is always well-formed (same pattern as src/eval/generation_metrics.py's
judge, which is a separate concern: this writes eval-set candidates, that
scores generated answers).
"""

from __future__ import annotations

import json

from src.generation.openrouter_client import get_openrouter_client

_CANDIDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_candidate",
        "description": "Submit a question the passage answers, and the answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["question", "answer"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You write evaluation questions for a RAG system being tested against "
    "this passage. Write ONE question that this passage fully answers on its "
    "own, plus the answer, grounded strictly in the passage — don't add "
    "anything the passage doesn't support. The question must stand alone: "
    "someone who has NOT read the passage, but who IS shown it as context, "
    "should be able to answer it. Avoid vague pronouns like 'it' or 'this "
    "document' that only make sense to someone already looking at the "
    "passage. Keep the answer concise and factual."
)


class CandidateGenerator:
    """Generates one (question, answer) pair per chunk of text via OpenRouter."""

    def __init__(self, model: str = "openai/gpt-4o-mini") -> None:
        self.model = model
        self._client = get_openrouter_client()

    def generate(self, chunk_text: str) -> tuple[str, str]:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            tools=[_CANDIDATE_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_candidate"}},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n\n{chunk_text}"},
            ],
        )
        tool_call = response.choices[0].message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)
        return data["question"], data["answer"]
