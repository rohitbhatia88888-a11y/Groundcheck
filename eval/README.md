# Golden set

`golden_set.json` is **not created yet**. Per `CLAUDE.md` it is **frozen once
an entry exists**: never edit or reorder an existing entry to make scores
look better. The only sanctioned writer is `GoldenSet.append_item` (see
`src/eval/golden_set.py`) — it only ever adds, and raises rather than
overwriting if an id collides.

## Building it: sample → review → append

Put real source PDFs in `data/raw/` first. Then:

```bash
# 1. Sample chunks stratified across documents, generate a candidate
#    (question, answer) pair for each via an LLM -> eval/candidates.jsonl.
#    Safe to re-run: never resamples a chunk_id already on record.
uv run python -m src.eval.sample_candidates --config configs/baseline.yaml -n 20

# 2. Review each pending candidate in the terminal: accept / edit / reject / skip.
#    Accepted pairs are appended to eval/golden_set.json immediately.
uv run python -m src.eval.review_candidates
```

`eval/candidates.jsonl` is working state (freely rewritten as you review —
see `src/eval/candidates.py`), unlike `golden_set.json`.

The candidate generator only ever produces `question_type: "simple"`
(one sampled chunk -> one question it alone answers) — see
`src/eval/candidate_generation.py`. Hard cases don't come from sampling one
chunk in isolation, so add them by hand, either by editing `golden_set.json`
directly or by overriding `question_type` at review time (the review loop
prompts for it on every accept).

## Schema (see `src/eval/golden_set.py`)

```json
{
  "items": [
    {
      "id": "cand-example-p3-c0",
      "question": "What does X do?",
      "relevant_chunks": [
        { "doc_id": "some-source-file", "page": 3 }
      ],
      "expected_answer": "A human-written reference answer.",
      "question_type": "simple"
    }
  ]
}
```

- `doc_id` is the parsed document's id — currently the source filename stem
  (see `PyMuPDFParser`).
- `relevant_chunks` is matched by `(doc_id, page)`, not `chunk_id` — chunk
  boundaries move whenever the chunking strategy changes, but the source page
  a fact lives on doesn't. This is what lets one golden set grade every
  experiment config in `configs/` without being rewritten per chunker.
- `expected_answer` is the reference used by the faithfulness/relevancy judge
  in `src/eval/generation_metrics.py`.
- `question_type`: `simple` | `multi_hop` | `unanswerable` | `distractor`.
  - `simple`: one chunk fully answers it (what the auto-generator produces).
  - `multi_hop`: needs several `relevant_chunks`, possibly across documents.
  - `unanswerable`: no chunk actually supports an answer — `relevant_chunks`
    is typically `[]`, and `expected_answer` should say the context doesn't
    contain it.
  - `distractor`: a plausible-looking but wrong chunk exists alongside the
    real one, testing whether retrieval/generation gets fooled by surface
    similarity.

Once populated, run an experiment against it with:

```python
from src.eval import run_experiment
run_experiment("configs/baseline.yaml")
```
