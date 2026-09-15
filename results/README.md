# Results

Written by `src/eval/runner.py` (`run_experiment`, or `python -m src.eval configs/baseline.yaml`).
Both gitignored (see `.gitignore`) — regenerate by re-running, don't hand-edit.

## `experiments.csv`

One row per experiment **run** (not per question) — appended to, never
rewritten, so every config you've ever run stays comparable in one table.
Columns, strictly separated per category (see `CLAUDE.md`):

- **Identity**: `config_name`, `config_path`, `timestamp`, `golden_set_hash`,
  `num_questions`, `chunker_type`, `embedder_model`, `reranker_type`,
  `generator_model`, `judge_model`
- **Retrieval**: `recall_at_1/3/5/10`, `mrr` — averaged only over questions
  with a real relevant chunk (`num_retrieval_eligible`); `null` if none exist
- **Generation**: `faithfulness`, `answer_relevance`
- **Behaviour**: `refusal_accuracy`, `num_unanswerable` — `null` if the
  golden set has no `question_type: "unanswerable"` items
- **Ops**: `latency_p50_seconds`, `latency_p95_seconds`, `cost_per_query_usd`
  (`null` if the provider never reported cost)

## `runs/<config_name>_<timestamp>.jsonl`

One row per golden-set question for that run — the retrieved chunks, the
generated answer, every judge's score *and reasoning*, latency, cost. This
is where to look when `experiments.csv` says a number is bad and you need to
know which question and why.
