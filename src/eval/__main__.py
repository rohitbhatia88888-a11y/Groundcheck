"""CLI entry point: `python -m src.eval configs/baseline.yaml`.

Runs the full golden set against the given config and prints a summary of
the four strictly-separate metric categories. Full detail: the aggregate
row is appended to results/experiments.csv, per-question detail goes to
results/runs/<config_name>_<timestamp>.jsonl (see src/eval/runner.py).
"""

from __future__ import annotations

import sys

from src.eval.runner import run_experiment


def _format_result(result) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    def usd(x: float | None) -> str:
        return "n/a" if x is None else f"${x:.5f}"

    def num(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    lines = [
        f"\n{'=' * 60}",
        f"Experiment: {result.config_name}  ({result.num_questions} questions)",
        f"Golden set hash: {result.golden_set_hash[:12]}...",
        (f"Chunker={result.chunker_type}  Embedder={result.embedder_model}  "
        f"Reranker={result.reranker_type}  Generator={result.generator_model}"),
        f"{'-' * 60}",
        "RETRIEVAL",
        (f"  recall@1={pct(result.recall_at_1)}  recall@3={pct(result.recall_at_3)}  "
        f"recall@5={pct(result.recall_at_5)}  recall@10={pct(result.recall_at_10)}"
        f"  ({result.num_retrieval_eligible} retrieval-eligible question(s))"),
        f"  MRR={num(result.mrr)}",
        "GENERATION",
        f"  faithfulness={result.faithfulness:.3f}  answer_relevance={result.answer_relevance:.3f}",
        f"  citation_validity_rate={pct(result.citation_validity_rate)}",
        "BEHAVIOUR",
        f"  refusal_accuracy={pct(result.refusal_accuracy)}  ({result.num_unanswerable} unanswerable question(s))",
        "OPS",
        (f"  latency p50={result.latency_p50_seconds:.2f}s  p95={result.latency_p95_seconds:.2f}s  "
        f"cost/query={usd(result.cost_per_query_usd)}"),
        f"{'-' * 60}",
        f"Per-question detail: {result.runs_path}",
        "Appended to: results/experiments.csv",
        "=" * 60,
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.eval <path-to-experiment.yaml>", file=sys.stderr)
        raise SystemExit(2)

    result = run_experiment(sys.argv[1])
    print(_format_result(result))


if __name__ == "__main__":
    main()
