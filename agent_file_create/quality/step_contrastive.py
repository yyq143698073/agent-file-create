"""Contrastive claim verification step — check "X优于Y" type claims."""

import logging

from agent_file_create.quality.step import QualityContext, QualityStep, StepResult

logger = logging.getLogger(__name__)


class ContrastiveStep(QualityStep):
    """Verify contrastive claims (e.g., 'A outperforms B') against source materials.

    Delegates to _verify_contrastive_claims in document._quality.
    """

    name = "contrastive"

    def run(self, ctx: QualityContext) -> StepResult:
        content = ctx.content
        analysis_results = ctx.analysis_results or []
        task_id = ctx.task_id

        try:
            from agent_file_create.document._quality import _verify_contrastive_claims

            source_text = "\n".join(
                str(ar.get("summary", "")) for ar in analysis_results[:5]
            )

            result = _verify_contrastive_claims(
                str(content or ""), source_text, task_id=str(task_id),
            )

            if result.get("flagged_count"):
                logger.info(
                    "contrastive_verify_summary flagged=%d/%d",
                    result.get("flagged_count", 0),
                    result.get("total_count", 0),
                )

            return StepResult(
                success=True,
                data={
                    "contrastive": result,
                    "flagged_count": result.get("flagged_count", 0),
                    "total_count": result.get("total_count", 0),
                },
            )

        except Exception as _e:
            logger.warning("contrastive_verify_failed err=%s", str(_e)[:200])
            return StepResult(success=False, error=str(_e))
