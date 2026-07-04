"""Quality check pipeline — decomposable, independently testable quality steps.

Usage:
    from agent_file_create.quality import QualityPipeline, QualityContext, QualityResult

    ctx = QualityContext(content="...", analysis_results=[...], task_id="...", output_dir="...")
    pipeline = QualityPipeline()
    result: QualityResult = pipeline.run(ctx)       # sequential
    # or
    result: QualityResult = pipeline.run_parallel(ctx)  # parallel
    print(result.to_dict())  # JSON-compatible report
"""

from agent_file_create.quality.step import QualityContext, StepResult, QualityStep
from agent_file_create.quality.pipeline import QualityPipeline, QualityResult

__all__ = [
    "QualityContext",
    "StepResult",
    "QualityStep",
    "QualityPipeline",
    "QualityResult",
]
