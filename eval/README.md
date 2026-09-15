# Golden set

`golden_set.json` is **not created yet**. It has to be authored against your
real documents in `data/raw/` with real, human-checked reference answers — it
isn't something to generate synthetically, and per `CLAUDE.md` it is **frozen
once created**: never edit it to make scores look better.

Schema (see `src/eval/golden_set.py`):

```json
{
  "items": [
    {
      "id": "q1",
      "question": "What does X do?",
      "relevant_chunks": [
        { "doc_id": "some-source-file", "page": 3 }
      ],
      "expected_answer": "A human-written reference answer."
    }
  ]
}
```

Notes:

- `doc_id` is the parsed document's id — currently the source filename stem
  (see `PyMuPDFParser`).
- `relevant_chunks` is matched by `(doc_id, page)`, not `chunk_id` — chunk
  boundaries move whenever the chunking strategy changes, but the source page
  a fact lives on doesn't. This is what lets one golden set grade every
  experiment config in `configs/` without being rewritten per chunker.
- `expected_answer` is the reference used by the faithfulness/relevancy judge
  in `src/eval/generation_metrics.py`.

Once real documents exist in `data/raw/` and this file is populated, run an
experiment with:

```python
from src.eval import run_experiment
run_experiment("configs/baseline.yaml")
```
