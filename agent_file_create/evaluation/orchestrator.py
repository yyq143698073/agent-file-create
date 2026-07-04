"""Evaluation orchestrator — combines Approach C (decomposed) + A (LLM judge).

Usage::

    from agent_file_create.evaluation.orchestrator import evaluate, evaluate_by_section

    # Full-document evaluation
    report = evaluate(
        content=state["content"],
        outline=state["outline"],
        analysis_results=state["analysis_results"],
        user_prompt=state["user_prompt"],
    )

    # Per-section breakdown
    sections = evaluate_by_section(content=state["content"], ...)
    for sec in sections:
        print(f"{sec['title']}: faith={sec['scores'].faithfulness:.2f}")
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from agent_file_create.evaluation.decomposed import run_decomposed_eval
from agent_file_create.evaluation.llm_judge import run_llm_judge
from agent_file_create.evaluation.models import DimensionScores, EvalReport

logger = logging.getLogger(__name__)

# Weight of LLM judge vs decomposed in combined score (0.0–1.0)
_LLM_WEIGHT = 0.6  # Judge weighs more since it catches semantic issues


def _split_by_sections(content: str) -> list[tuple[str, str]]:
    """Split document content by ## headings, returning [(title, body), ...]."""
    if not content or not content.strip():
        return []

    lines = content.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "（前言）"
    current_body: list[str] = []

    for line in lines:
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))

    return sections


def _weighted_combine(a: DimensionScores, b: DimensionScores, w_b: float = _LLM_WEIGHT) -> DimensionScores:
    """Weighted average of two score sets."""
    w_a = 1.0 - w_b
    return DimensionScores(
        relevance=round(a.relevance * w_a + b.relevance * w_b, 3),
        faithfulness=round(a.faithfulness * w_a + b.faithfulness * w_b, 3),
        coherence=round(a.coherence * w_a + b.coherence * w_b, 3),
        completeness=round(a.completeness * w_a + b.completeness * w_b, 3),
    )


def evaluate(
    content: str,
    outline: str = "",
    analysis_results: Optional[List[dict]] = None,
    user_prompt: str = "",
    *,
    enable_llm: bool = True,
    llm_weight: float = _LLM_WEIGHT,
    **llm_kwargs,
) -> EvalReport:
    """Run both evaluation approaches and return a combined report.

    Parameters
    ----------
    content:
        The generated document content to evaluate.
    outline:
        The markdown outline that was used for generation.
    analysis_results:
        Source material extraction results (list of dict per file).
    user_prompt:
        The original user request / topic.
    enable_llm:
        Set to ``False`` to skip LLM judge (decomposed metrics only).
    llm_weight:
        Weight of LLM judge scores in combined result (0–1, default 0.6).
    **llm_kwargs:
        Passed through to :func:`run_llm_judge` (model_name, api_style, etc.).

    Returns
    -------
    EvalReport
        Containing decomposed, llm_judge, and combined scores plus metadata.
    """
    analysis_results = analysis_results or []
    report = EvalReport()

    # ── Approach C: Decomposed / rule‑based ───────────────────────────
    logger.info("Running decomposed evaluation ...")
    try:
        scores_c, details = run_decomposed_eval(
            content=content,
            outline=outline,
            analysis_results=analysis_results,
            user_prompt=user_prompt,
        )
        report.decomposed = scores_c
        report.decomposed_details = details
        report.warnings.extend(details.get("faithfulness_warnings", []))
    except Exception as exc:
        logger.warning("Decomposed eval failed: %s", exc)
        report.warnings.append(f"规则评估失败: {exc}")

    # ── Approach A: LLM‑as‑Judge ──────────────────────────────────────
    if enable_llm:
        logger.info("Running LLM judge evaluation ...")
        try:
            scores_a, reasoning = run_llm_judge(
                content=content,
                analysis_results=analysis_results,
                user_prompt=user_prompt,
                **llm_kwargs,
            )
            report.llm_judge = scores_a
            report.llm_reasoning = reasoning
        except Exception as exc:
            logger.warning("LLM judge failed: %s", exc)
            report.warnings.append(f"LLM评估失败: {exc}")
    else:
        logger.info("LLM judge disabled.")

    # ── Combined score ────────────────────────────────────────────────
    if enable_llm and report.llm_judge.relevance > 0:
        report.combined = _weighted_combine(report.decomposed, report.llm_judge, llm_weight)
    else:
        report.combined = report.decomposed

    # Summary
    avg = (
        report.combined.relevance
        + report.combined.faithfulness
        + report.combined.coherence
        + report.combined.completeness
    ) / 4.0
    logger.info(
        "Eval done — combined: rel=%.2f faith=%.2f coh=%.2f comp=%.2f (avg=%.2f)",
        report.combined.relevance,
        report.combined.faithfulness,
        report.combined.coherence,
        report.combined.completeness,
        avg,
    )

    return report


def evaluate_by_section(
    content: str,
    outline: str = "",
    analysis_results: Optional[List[dict]] = None,
    user_prompt: str = "",
    *,
    enable_llm: bool = False,
) -> list[dict]:
    """Evaluate each section of the document independently.

    Splits content by ``##`` headings and runs decomposed evaluation on each
    section body. Returns a list of per-section score dicts, sorted by
    faithfulness ascending (weakest sections first).

    LLM judge is disabled by default (cost/performance) — set enable_llm=True
    for per-section LLM scoring.

    Returns
    -------
    list[dict]
        Each dict: {"title": str, "chars": int, "scores": DimensionScores,
                    "warnings": list[str]}
    """
    analysis_results = analysis_results or []
    sections = _split_by_sections(content)
    if not sections:
        return []

    results: list[dict] = []
    for title, body in sections:
        if len(body) < 50:
            continue  # Skip very short sections
        try:
            scores_c, details = run_decomposed_eval(
                content=body,
                outline=outline,
                analysis_results=analysis_results,
                user_prompt=user_prompt,
            )
            section_result: dict = {
                "title": title,
                "chars": len(body),
                "scores": scores_c,
                "warnings": details.get("faithfulness_warnings", []),
            }
            if enable_llm:
                try:
                    scores_a, _ = run_llm_judge(
                        content=body,
                        analysis_results=analysis_results,
                        user_prompt=user_prompt,
                    )
                    section_result["scores"] = _weighted_combine(scores_c, scores_a)
                    section_result["llm_judge"] = scores_a
                except Exception as exc:
                    logger.debug("section_llm_judge_failed section=%s err=%s", title, exc)
            results.append(section_result)
        except Exception as exc:
            logger.debug("section_eval_failed section=%s err=%s", title, exc)

    # Sort by faithfulness ascending — weakest sections first
    results.sort(key=lambda r: r["scores"].faithfulness)
    return results
