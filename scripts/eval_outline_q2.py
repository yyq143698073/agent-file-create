"""Q2 outline quality evaluation using PubMed-RCT.

Metrics:
- Structure pass rate: _validate_outline passes (H1/H2/H3/no skip)
- Naming emptiness rate: fraction of headings flagged as template-like
- Critical section rate: fraction with a valid critical chapter
- Topic coverage: jieba keyword match between user prompt and outline

Usage:
  python scripts/eval_outline_q2.py --sample 50 --output result/q2_eval_v2_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# PubMed-RCT label mapping
_LABEL_MAP = {0: "BACKGROUND", 1: "OBJECTIVE", 2: "METHODS", 3: "RESULTS", 4: "CONCLUSIONS"}


def _load_pubmed_rct(sample_count: int) -> list[dict]:
    """Load PubMed-RCT test split and group sentences into abstracts."""
    from datasets import load_dataset

    ds = load_dataset("armanc/pubmed-rct20k", split="test")
    abstracts: dict[str, list[dict]] = {}
    for item in ds:
        aid = str(item["abstract_id"])
        if aid not in abstracts:
            abstracts[aid] = []
        label_raw = item["label"]
        if isinstance(label_raw, int):
            label_name = _LABEL_MAP.get(label_raw, str(label_raw))
        else:
            label_name = str(label_raw).upper()
        abstracts[aid].append({
            "sentence_id": int(item["sentence_id"]),
            "text": str(item["text"] or "").strip(),
            "label_name": label_name,
        })

    # Sort sentences, reconstruct full text
    result: list[dict] = []
    for aid, sentences in list(abstracts.items())[:sample_count]:
        sentences.sort(key=lambda x: x["sentence_id"])
        full_text = " ".join(s["text"] for s in sentences)
        # Extract user prompt from the first sentence + label structure
        sections_present = sorted(set(s["label_name"] for s in sentences))
        user_prompt = (
            f"请基于以下生物医学摘要撰写一份结构化报告。摘要包含以下章节类型："
            f"{'、'.join(sections_present)}。"
        )
        result.append({
            "abstract_id": aid,
            "sentence_count": len(sentences),
            "full_text": full_text,
            "user_prompt": user_prompt,
            "expected_sections": sections_present,
        })
    return result


def _evaluate_single(item: dict, idx: int) -> dict:
    """Generate outline for one abstract and evaluate quality."""
    from agent_file_create.document.outline_generator import (
        generate_outline,
        _validate_outline,
        _check_naming_quality,
        _check_critical_section,
        _check_topic_coverage,
        _TEMPLATE_TITLE_PATTERNS,
    )

    # Build mock multimodal_results from the abstract text
    multimodal_results = {
        f"abstract_{idx}": {
            "title": f"PubMed Abstract {item['abstract_id']}",
            "summary": item["full_text"][:500],
            "key_points": [item["full_text"][i:i + 200] for i in range(0, min(600, len(item["full_text"])), 200)],
        }
    }

    t0 = time.time()
    try:
        outline = generate_outline(multimodal_results, item["user_prompt"])
    except Exception as e:
        return {"abstract_id": item["abstract_id"], "error": str(e)[:200], "seconds": round(time.time() - t0, 2)}
    elapsed = round(time.time() - t0, 2)

    # ── Evaluate ──
    issues = _validate_outline(outline)
    naming_warnings = _check_naming_quality(outline)
    critical_issues = _check_critical_section(outline)
    topic_cov = _check_topic_coverage(outline, item["user_prompt"])

    # Count template-like headings
    headings = re.findall(r"^#{2,3}\s+(.+)$", outline, re.MULTILINE)
    total_headings = len(headings)
    template_count = 0
    for h in headings:
        clean = re.sub(r"^[\d.]+\s*", "", h).strip()
        for pat in _TEMPLATE_TITLE_PATTERNS:
            if re.search(pat, clean) and len(clean) <= 8:
                template_count += 1
                break

    naming_emptiness = template_count / max(total_headings, 1)

    # H2 count
    h2_count = len(re.findall(r"^##\s", outline, re.MULTILINE))
    h3_count = len(re.findall(r"^###\s", outline, re.MULTILINE))

    return {
        "abstract_id": item["abstract_id"],
        "structure_pass": not issues,
        "structure_issues": issues,
        "h2_count": h2_count,
        "h3_count": h3_count,
        "total_headings": total_headings,
        "template_headings": template_count,
        "naming_emptiness_rate": round(naming_emptiness, 4),
        "naming_warnings": naming_warnings,
        "critical_section_ok": not critical_issues,
        "critical_issues": critical_issues,
        "topic_coverage": topic_cov["coverage"],
        "topic_uncovered": topic_cov.get("uncovered", []),
        "outline_preview": outline[:500],
        "seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q2 outline quality evaluation")
    parser.add_argument("--sample", type=int, default=20, help="Number of PubMed abstracts (default 20)")
    parser.add_argument("--output", default="result/q2_eval_v2_baseline.json")
    args = parser.parse_args()

    print("=" * 50)
    print("Q2 Outline Quality Evaluation — PubMed-RCT")
    print(f"  Samples: {args.sample}")
    print()

    # 1. Load data
    print("[1/3] Loading PubMed-RCT ...")
    samples = _load_pubmed_rct(args.sample)
    print(f"  Loaded {len(samples)} abstracts")

    # 2. Evaluate
    print(f"[2/3] Generating outlines ({len(samples)} samples) ...")
    results: list[dict] = []
    t0 = time.time()
    for idx, item in enumerate(samples):
        print(f"  [{idx + 1}/{len(samples)}] {item['abstract_id']} ...", end=" ", flush=True)
        r = _evaluate_single(item, idx)
        results.append(r)
        status = "PASS" if r.get("structure_pass") else f"FAIL({len(r.get('structure_issues', []))} issues)"
        naming = r.get("naming_emptiness_rate", 0)
        crit = "OK" if r.get("critical_section_ok") else "MISS"
        secs = r.get("seconds", 0)
        print(f"{status} naming={naming:.0%} crit={crit} {secs}s")

    elapsed = time.time() - t0

    # 3. Summarize
    import statistics

    pass_count = sum(1 for r in results if r.get("structure_pass"))
    naming_rates = [r.get("naming_emptiness_rate", 0) for r in results]
    crit_ok = sum(1 for r in results if r.get("critical_section_ok"))
    topic_covs = [r.get("topic_coverage", 0) for r in results]
    avg_secs = round(statistics.mean([r.get("seconds", 0) for r in results]), 2)

    print()
    print("=" * 50)
    print(f"Structure Pass Rate:   {pass_count}/{len(results)} ({pass_count / max(len(results), 1):.1%})")
    print(f"Naming Emptiness Rate: {statistics.mean(naming_rates):.1%} ± {statistics.pstdev(naming_rates):.1%}")
    print(f"Critical Section Rate: {crit_ok}/{len(results)} ({crit_ok / max(len(results), 1):.1%})")
    print(f"Topic Coverage:        {statistics.mean(topic_covs):.1%} ± {statistics.pstdev(topic_covs):.1%}")
    print(f"Avg time/sample:       {avg_secs}s")
    print(f"Total time:            {elapsed:.0f}s")
    print()

    # Save
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset": "armanc/pubmed-rct20k",
            "sample_count": len(results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_seconds": round(elapsed, 2),
        },
        "summary": {
            "structure_pass_rate": pass_count / max(len(results), 1),
            "naming_emptiness_rate": statistics.mean(naming_rates),
            "naming_emptiness_std": statistics.pstdev(naming_rates),
            "critical_section_rate": crit_ok / max(len(results), 1),
            "topic_coverage_mean": statistics.mean(topic_covs),
            "avg_seconds": avg_secs,
        },
        "details": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
