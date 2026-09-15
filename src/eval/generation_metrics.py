"""LLM-as-judge generation metrics: faithfulness and answer relevancy.

Judges via a forced tool call so scores are always well-formed, not free text
to parse. Kept in a separate module from retrieval_metrics.py by design — see
CLAUDE.md: retrieval and generation metrics are computed and reported
separately, never blended into one score.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from src.generation.models import Answer
from src.generation.openrouter_client import get_openrouter_client

_SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_score",
        "description": "Submit a numeric judgment with reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        },
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


class OpenRouterJudge:
    """LLM-as-judge for generation-quality metrics, backed by OpenRouter."""

    def __init__(self, model: str = "openai/gpt-4o-mini") -> None:
        self.model = model
        self._client = get_openrouter_client()

    def _judge(self, system_prompt: str, user_message: str) -> JudgeScore:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            tools=[_SCORE_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_score"}},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return JudgeScore.model_validate(json.loads(tool_call.function.arguments))

    def score_faithfulness(self, answer: Answer) -> JudgeScore:
        context_block = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in answer.context)
        user_message = f"Context:\n\n{context_block}\n\nAnswer to grade:\n{answer.text}"
        return self._judge(_FAITHFULNESS_PROMPT, user_message)

    def score_relevancy(self, question: str, answer: Answer) -> JudgeScore:
        user_message = f"Question: {question}\n\nAnswer to grade:\n{answer.text}"
        return self._judge(_RELEVANCY_PROMPT, user_message)
