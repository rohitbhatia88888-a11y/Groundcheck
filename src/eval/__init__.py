from src.eval.candidate_generation import CandidateGenerator
from src.eval.candidates import Candidate, load_candidates, save_candidates
from src.eval.config import ComponentConfig, ExperimentConfig
from src.eval.generation_metrics import JudgeScore, OpenRouterJudge
from src.eval.golden_set import GoldenSet, GoldenSetItem, QuestionType, RelevantChunkRef
from src.eval.registry import Pipeline, build_pipeline
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
    "GoldenSet",
    "GoldenSetItem",
    "JudgeScore",
    "OpenRouterJudge",
    "Pipeline",
    "QuestionType",
    "RelevantChunkRef",
    "build_pipeline",
    "ingest_raw_documents",
    "load_candidates",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "run_experiment",
    "save_candidates",
]
