from src.eval.behaviour_metrics import refusal_accuracy
from src.eval.candidate_generation import CandidateGenerator
from src.eval.candidates import Candidate, load_candidates, save_candidates
from src.eval.config import ComponentConfig, ExperimentConfig
from src.eval.generation_metrics import JudgeScore, OpenRouterJudge, RefusalJudgment
from src.eval.golden_set import (
    GoldenSet,
    GoldenSetItem,
    QuestionType,
    RelevantChunkRef,
    verify_golden_set_hash,
)
from src.eval.ops_metrics import mean_cost_usd, percentile
from src.eval.registry import Pipeline, build_pipeline
from src.eval.results import ExperimentResult, QuestionResult, RetrievedRef
from src.eval.retrieval_metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from src.eval.runner import ingest_raw_documents, run_experiment

# NOT re-exported here: review_candidates.run_review, sample_candidates.*.
# Both those modules are run directly via `python -m src.eval.<name>`; eagerly
# importing them into this __init__ makes runpy re-import them under
# __main__, which triggers "found in sys.modules ... prior to execution"
# RuntimeWarnings on every invocation. Import them from their own module path
# instead (`from src.eval.review_candidates import run_review`).

__all__ = [
    "Candidate",
    "CandidateGenerator",
    "ComponentConfig",
    "ExperimentConfig",
    "ExperimentResult",
    "GoldenSet",
    "GoldenSetItem",
    "JudgeScore",
    "OpenRouterJudge",
    "Pipeline",
    "QuestionResult",
    "QuestionType",
    "RefusalJudgment",
    "RelevantChunkRef",
    "RetrievedRef",
    "build_pipeline",
    "ingest_raw_documents",
    "load_candidates",
    "mean_cost_usd",
    "mean_reciprocal_rank",
    "percentile",
    "precision_at_k",
    "recall_at_k",
    "refusal_accuracy",
    "run_experiment",
    "save_candidates",
    "verify_golden_set_hash",
]
