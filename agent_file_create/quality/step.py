"""Base classes for quality check pipeline steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class QualityContext:
    """Immutable context passed to each quality check step.

    Steps should read from this context but not modify it.
    Results are returned via StepResult.
    """

    content: str
    analysis_results: list[dict]
    task_id: str
    output_dir: str


@dataclass
class StepResult:
    """Structured result from a single quality check step.

    Attributes:
        success: Whether the step completed without error.
        data: Arbitrary step-specific output data (warnings, metrics, etc.).
        error: Human-readable error message if success is False.
        warnings: Non-fatal warnings collected during the step.
        content: Optionally modified content (e.g., after re-retrieval fixes).
    """

    success: bool
    data: dict = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    content: str | None = None   # If the step modified content


class QualityStep(ABC):
    """Abstract base class for a quality check step.

    Each step performs one focused quality check and returns a StepResult.
    Steps should be stateless — all context comes from QualityContext.
    """

    name: str = "base"

    @abstractmethod
    def run(self, ctx: QualityContext) -> StepResult:
        """Execute this quality check step.

        Args:
            ctx: Immutable context with content, analysis_results, etc.

        Returns:
            StepResult with success flag, any data/warnings, and optionally modified content.
        """
        ...
