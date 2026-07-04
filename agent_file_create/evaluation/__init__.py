"""Document generation evaluation — Approach C (decomposed) + A (LLM judge)."""

from agent_file_create.evaluation.models import DimensionScores, EvalReport
from agent_file_create.evaluation.orchestrator import evaluate

__all__ = ["evaluate", "EvalReport", "DimensionScores"]
