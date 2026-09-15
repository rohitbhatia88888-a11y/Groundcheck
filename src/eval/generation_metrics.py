"""LLM-as-judge generation metrics: faithfulness and answer relevancy.

Judges via a forced tool call so scores are always well-formed, not free text
to parse. Kept in a separate module from retrieval_metrics.py by design — see
CLAUDE.md: retrieval and generation metrics are computed and reported
separately, never blended into one score.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel

from src.generation.models import Answer

_SCORE_TOOL = {
    "name": "submit_score",
    "description": "Submit a numeric judgment with reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["score", "reasoning"],
    },
}

_FAITHFULNESS_PROMPT = (
    "You are grading whether an answer is faithful to its source context — i.e. "
    "every claim in the answer is actually supported by the context, with no "
    "unsupported additions. Score 1.0 if fully supported, 0.0 if entirely "
    "unsupported, and a value in between for partial support. Judge "
    "faithfulness only — do not penalize for the answer being incomplete or "
    "for style."
)

_RELEVANCY_PROMPT = (
    "You are grading whether an answer actually addresses the question asked. "
    "Score 1.0 if it directly and fully addresses the question, 0.0 if it is "
    "off-topic or non-responsive, and a value in between for partial "
    "relevance. Judge relevancy only — do not penalize for factual accuracy."
)


class JudgeScore(BaseModel):
    score: float
    reasoning: str


class ClaudeJudge:
    """LLM-as-judge for generation-quality metrics, backed by the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self._client = anthropic.Anthropic()

    def _judge(self, system_prompt: str, user_message: str) -> JudgeScore:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system_prompt,
            tools=[_SCORE_TOOL],
            tool_choice={"type": "tool", "name": "submit_score"},
            messages=[{"role": "user", "content": user_message}],
        )
        tool_use = next(block for block in response.content if block.type == "tool_use")
        return JudgeScore.model_validate(tool_use.input)

    def score_faithfulness(self, answer: Answer) -> JudgeScore:
        context_block = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in answer.context)
        user_message = f"Context:\n\n{context_block}\n\nAnswer to grade:\n{answer.text}"
        return self._judge(_FAITHFULNESS_PROMPT, user_message)

    def score_relevancy(self, question: str, answer: Answer) -> JudgeScore:
        user_message = f"Question: {question}\n\nAnswer to grade:\n{answer.text}"
        return self._judge(_RELEVANCY_PROMPT, user_message)
