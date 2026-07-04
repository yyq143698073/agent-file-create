"""FActScore + Coverage step — factual accuracy and aspect coverage evaluation."""

import logging

from agent_file_create.quality.step import QualityContext, QualityStep, StepResult

logger = logging.getLogger(__name__)


class FactscoreStep(QualityStep):
    """Compute FActScore and aspect coverage metrics.

    Optionally fills coverage gaps via KB search when coverage < 0.8.
    """

    name = "factscore"

    def run(self, ctx: QualityContext) -> StepResult:
        content = ctx.content
        analysis_results = ctx.analysis_results or []
        task_id = ctx.task_id
        output_dir = ctx.output_dir

        try:
            from agent_file_create.document._quality import (
                _compute_factscore_and_coverage,
                _fill_coverage_gaps,
            )

            # Phase 1: Compute metrics
            coverage_results = _compute_factscore_and_coverage(
                str(content or ""), analysis_results, task_id=str(task_id),
            )

            # Phase 2: Fill coverage gaps if needed
            if (coverage_results.get("coverage") or 1.0) < 0.8 and coverage_results.get("uncovered_aspects"):
                try:
                    filled = _fill_coverage_gaps(
                        str(content or ""),
                        coverage_results["uncovered_aspects"],
                        analysis_results,
                        task_id=str(task_id),
                    )
                    if filled != content:
                        content = filled
                        try:
                            from pathlib import Path
                            (Path(output_dir) / "content.md").write_text(content, encoding="utf-8")
                        except Exception as e:
                            logger.debug("factscore coverage write failed: %s", e)
                except Exception as _e:
                    logger.warning("coverage_gap_fill_failed err=%s", str(_e)[:200])

            return StepResult(
                success=True, content=content,
                data={
                    "factscore": coverage_results.get("factscore"),
                    "coverage": coverage_results.get("coverage"),
                    "facts_verified": coverage_results.get("verified_count", 0),
                    "facts_total": coverage_results.get("facts_count", 0),
                    "aspects_covered": coverage_results.get("covered_count", 0),
                    "aspects_total": coverage_results.get("aspects_count", 0),
                    "uncovered_aspects": coverage_results.get("uncovered_aspects", []),
                },
            )

        except Exception as _e:
            logger.warning("factscore_coverage_failed err=%s", str(_e)[:200])
            return StepResult(success=False, error=str(_e))
