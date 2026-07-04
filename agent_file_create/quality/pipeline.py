"""Quality pipeline orchestrator — runs quality steps sequentially or in parallel."""

from __future__ import annotations

import logging
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_file_create.quality.step import QualityContext, StepResult, QualityStep
from agent_file_create.quality.step_faithfulness import FaithfulnessStep
from agent_file_create.quality.step_citation import CitationStep
from agent_file_create.quality.step_contrastive import ContrastiveStep
from agent_file_create.quality.step_factscore import FactscoreStep

logger = logging.getLogger(__name__)

# ── Feature flag for parallel execution ──
_QUALITY_PARALLEL_ENABLED = os.getenv("QUALITY_PARALLEL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}


class QualityPipeline:
    """Orchestrator for quality check steps.

    Supports both sequential (run) and parallel (run_parallel) execution.
    Content flows through steps — each step may modify content via StepResult.content.

    Usage:
        pipeline = QualityPipeline()
        result = pipeline.run(ctx)            # sequential
        result = pipeline.run_parallel(ctx)   # parallel (env=QUALITY_PARALLEL_ENABLED)
    """

    def __init__(self, steps: list[QualityStep] | None = None):
        """Initialize pipeline with optional custom step list.

        Args:
            steps: List of QualityStep instances. If None, uses default steps.
        """
        self._steps = steps or [
            FaithfulnessStep(),
            CitationStep(),
            ContrastiveStep(),
            FactscoreStep(),
        ]

    @property
    def steps(self) -> list[QualityStep]:
        """Read-only access to the step list."""
        return list(self._steps)

    def run(self, ctx: QualityContext) -> QualityResult:
        """Run all steps sequentially.

        Content flows through — each step's output content becomes the
        next step's input. Earlier steps' data is merged into the final result.
        """
        t0 = _time.perf_counter()
        content = ctx.content
        all_data: dict = {}
        all_warnings: list[str] = []
        step_results: list[StepResult] = []

        for step in self._steps:
            # Update context with potentially modified content
            step_ctx = QualityContext(
                content=content,
                analysis_results=ctx.analysis_results,
                task_id=ctx.task_id,
                output_dir=ctx.output_dir,
            )

            try:
                result = step.run(step_ctx)
            except Exception as e:
                logger.error("quality_step_failed step=%s err=%s", step.name, e)
                result = StepResult(success=False, error=str(e))

            step_results.append(result)

            if result.success and result.content is not None:
                content = result.content

            if result.data:
                all_data[step.name] = result.data
            if result.warnings:
                all_warnings.extend(result.warnings)

        return QualityResult(
            success=all(r.success for r in step_results),
            content=content,
            data=all_data,
            warnings=all_warnings,
            step_results=step_results,
            elapsed_ms=int((_time.perf_counter() - t0) * 1000),
        )

    def run_parallel(self, ctx: QualityContext) -> QualityResult:
        """Run all steps in parallel via ThreadPoolExecutor.

        Since steps are independent (no data dependency between them),
        they can execute concurrently. Results are merged afterward.
        Content is NOT piped between steps — each step sees the original content.
        """
        t0 = _time.perf_counter()
        if not _QUALITY_PARALLEL_ENABLED:
            logger.debug("quality_parallel disabled, falling back to sequential")
            return self.run(ctx)

        results: dict[str, StepResult] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(self._steps))) as pool:
            futures = {
                pool.submit(step.run, ctx): step.name
                for step in self._steps
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:
                    logger.error("quality_step_failed step=%s err=%s", name, e)
                    results[name] = StepResult(success=False, error=str(e))

        # Merge: last step's content wins (or original if none modified)
        step_results = [results.get(s.name, StepResult(success=False, error="missing"))
                        for s in self._steps]
        final_content = ctx.content
        for r in step_results:
            if r.success and r.content is not None and r.content != ctx.content:
                final_content = r.content

        all_data: dict = {}
        all_warnings: list[str] = []
        for name, r in results.items():
            if r.data:
                all_data[name] = r.data
            if r.warnings:
                all_warnings.extend(r.warnings)

        return QualityResult(
            success=all(r.success for r in step_results),
            content=final_content,
            data=all_data,
            warnings=all_warnings,
            step_results=step_results,
            elapsed_ms=int((_time.perf_counter() - t0) * 1000),
        )


class QualityResult:
    """Aggregated result from a quality pipeline run.

    Attributes:
        success: Whether all steps completed without errors.
        content: Final content after all step modifications (may differ from input).
        data: Per-step output data keyed by step name.
        warnings: All warnings collected across all steps.
        step_results: Individual StepResult objects in execution order.
        severity_counts: Breakdown of issues by severity level.
        metadata: Timing and context metadata.
        summary: Optional human-readable summary (can be set after construction).
    """

    def __init__(
        self,
        success: bool,
        content: str,
        data: dict,
        warnings: list[str],
        step_results: list[StepResult],
        *,
        elapsed_ms: int = 0,
        sections_processed: int = 0,
    ):
        self.success = success
        self.content = content
        self.data = data
        self.warnings = warnings
        self.step_results = step_results
        self.severity_counts: dict[str, int] = {}
        self.metadata: dict = {
            "elapsed_ms": elapsed_ms,
            "sections_processed": sections_processed,
            "steps_run": len(step_results),
            "steps_passed": sum(1 for r in step_results if r.success),
        }
        self.summary: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for evaluation reports."""
        return {
            "success": self.success,
            "warnings": self.warnings,
            "warnings_count": len(self.warnings),
            "severity_counts": self.severity_counts,
            "metadata": self.metadata,
            "summary": self.summary,
            "steps": [
                {
                    "success": r.success,
                    "warnings": r.warnings,
                    "error": r.error,
                }
                for r in self.step_results
            ],
        }

    def __repr__(self) -> str:
        return (f"QualityResult(success={self.success}, steps={len(self.step_results)}, "
                f"warnings={len(self.warnings)}, data_keys={list(self.data.keys())})")
