"""Data models for document evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


@dataclass
class DimensionScores:
    """Scores for one evaluation dimension (0.0–1.0 or 1–5 scale)."""
    relevance: float = 0.0
    faithfulness: float = 0.0
    coherence: float = 0.0
    completeness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "relevance": round(self.relevance, 3),
            "faithfulness": round(self.faithfulness, 3),
            "coherence": round(self.coherence, 3),
            "completeness": round(self.completeness, 3),
        }


@dataclass
class EvalReport:
    """Aggregated evaluation report from decomposed + LLM-judge approaches."""

    # Approach C: decomposed / rule-based scores
    decomposed: DimensionScores = field(default_factory=DimensionScores)

    # Approach A: LLM-as-judge scores (1-5 scale, normalized to 0-1)
    llm_judge: DimensionScores = field(default_factory=DimensionScores)

    # Combined weighted scores
    combined: DimensionScores = field(default_factory=DimensionScores)

    # Metadata
    llm_reasoning: str = ""           # Judge's explanation
    llm_judge_raw: str = ""           # Raw LLM response
    decomposed_details: dict = field(default_factory=dict)  # Per-metric details
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        combined_dict = self.combined.to_dict()
        return {
            "decomposed": self.decomposed.to_dict(),
            "llm_judge": self.llm_judge.to_dict(),
            "combined": combined_dict,
            "llm_reasoning": self.llm_reasoning[:500],
            "decomposed_details": self.decomposed_details,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "factscore": combined_dict.get("faithfulness"),
            "coverage": combined_dict.get("completeness"),
            "consistency_score": combined_dict.get("coherence"),
        }
