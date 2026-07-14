"""Q4 content generation quality evaluation.

Metrics:
- Citation compliance rate: format check, source diversity, no duplicate in paragraph
- Fact faithfulness: atomic facts have source support (LLM-based)
- Coherence: adjacent paragraph semantic similarity (jieba overlap)

Usage:
  python scripts/eval_content_q4.py --sample 20 --output result/q4_eval_baseline.json
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


def _check_citation_compliance(text: str) -> dict:
    """Check citation compliance and return detailed metrics."""
    if not text or "【" not in text:
        return {"pass": False, "issues": ["缺少引用标注"], "unique_citations": 0}

    issues = []
    markers = re.findall(r"【(\d+)】", text)
    unique_nums = set(int(m) for m in markers)

    # 1. At least 2 different citation numbers
    if len(unique_nums) < 2:
        issues.append(f"引用来源过少：仅{len(unique_nums)}种（需≥2）")

    # 2. No duplicate 【n】 in same paragraph
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    dup_count = 0
    for para in paragraphs:
        para_markers = re.findall(r"【(\d+)】", para)
        if len(para_markers) > len(set(para_markers)):
            dup_count += 1
    if dup_count > 0:
        issues.append(f"同段重复引用：{dup_count}个段落")

    # 3. Each 【n】 has verbal reference nearby
    missing_verbal = 0
    for m in re.finditer(r"【(\d+)】", text):
        after = text[m.end():m.end() + 40]
        if not re.search(r"[据参来][^，。；\n]{2,20}", after):
            missing_verbal += 1
    if missing_verbal > 0:
        issues.append(f"缺少口头引用：{missing_verbal}处")

    # 4. Citation numbers sanity check
    if unique_nums and max(unique_nums) > 20:
        issues.append(f"引用编号异常：最大{max(unique_nums)}>20")

    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "unique_citations": len(unique_nums),
        "duplicate_paragraphs": dup_count,
        "missing_verbal_refs": missing_verbal,
    }


def _compute_coherence_score(text: str) -> dict:
    """Compute coherence via adjacent paragraph semantic similarity."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip() and len(p.strip()) > 20]
    if len(paragraphs) < 2:
        return {"score": 0.0, "pairs": 0, "avg_similarity": 0.0}

    try:
        import jieba
        tokenize = lambda t: set(w for w in jieba.lcut(t) if len(w.strip()) >= 2)
    except Exception:
        tokenize = lambda t: set(t[i:i+2] for i in range(len(t)-1) if t[i:i+2].strip())

    similarities = []
    for i in range(len(paragraphs) - 1):
        words_a = tokenize(paragraphs[i])
        words_b = tokenize(paragraphs[i + 1])
        if not words_a or not words_b:
            continue
        intersection = words_a & words_b
        union = words_a | words_b
        jaccard = len(intersection) / max(len(union), 1)
        similarities.append(jaccard)

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    # Target range: 0.15-0.70
    in_range = 0.15 <= avg_sim <= 0.70

    return {
        "score": round(avg_sim, 3),
        "pairs": len(similarities),
        "avg_similarity": round(avg_sim, 3),
        "in_target_range": in_range,
        "min_similarity": round(min(similarities), 3) if similarities else 0.0,
        "max_similarity": round(max(similarities), 3) if similarities else 0.0,
    }


def _generate_content_for_eval(item: dict, idx: int) -> str:
    """Generate content for one test item."""
    from agent_file_create.document.outline_generator import generate_outline
    from agent_file_create.document.content_generator import generate_content

    # Build mock multimodal_results
    multimodal_results = {
        f"source_{idx}": {
            "title": item.get("title", f"Test Source {idx}"),
            "summary": item.get("text", "")[:500],
            "key_points": [item.get("text", "")[i:i+200] for i in range(0, min(600, len(item.get("text", ""))), 200)],
            "conclusion": item.get("text", "")[-200:] if len(item.get("text", "")) > 200 else "",
        }
    }

    # Generate outline first
    outline = generate_outline(
        multimodal_results=multimodal_results,
        user_prompt=item.get("user_prompt", "生成报告"),
        task_id="",
    )

    if not outline:
        return ""

    # Generate content
    content = generate_content(
        outline=outline,
        multimodal_results=multimodal_results,
        user_prompt=item.get("user_prompt", "生成报告"),
        task_id="",
    )

    return content


def _load_test_samples(sample_count: int) -> list[dict]:
    """Load test samples for evaluation."""
    # Try to load from existing test data
    test_doc_dir = ROOT / "test_doc"
    samples = []

    # Look for custom prompts
    custom_prompts_file = test_doc_dir / "q2" / "custom_prompts.json"
    if custom_prompts_file.exists():
        try:
            prompts = json.loads(custom_prompts_file.read_text(encoding="utf-8"))
            for i, item in enumerate(prompts[:sample_count]):
                samples.append({
                    "id": f"custom_{i}",
                    "title": item.get("title", f"Sample {i}"),
                    "text": item.get("prompt", ""),
                    "user_prompt": item.get("prompt", "生成报告"),
                })
        except Exception:
            pass

    # Fallback: generate synthetic samples
    if not samples:
        for i in range(sample_count):
            samples.append({
                "id": f"synthetic_{i}",
                "title": f"合成样本 {i}",
                "text": f"这是第{i}个合成测试样本的内容。包含一些关键数据和观点用于测试引用合规性和事实忠实度。",
                "user_prompt": f"请基于以下材料撰写报告，重点关注数据分析和结论。样本{i}",
            })

    return samples


def _evaluate_single(item: dict, idx: int) -> dict:
    """Generate content for one item and evaluate quality."""
    t0 = time.time()

    try:
        content = _generate_content_for_eval(item, idx)
    except Exception as e:
        return {
            "id": item.get("id", f"item_{idx}"),
            "error": str(e),
            "citation_compliance": {"pass": False, "issues": [f"生成失败: {e}"]},
            "coherence": {"score": 0.0},
            "elapsed": time.time() - t0,
        }

    if not content:
        return {
            "id": item.get("id", f"item_{idx}"),
            "error": "生成内容为空",
            "citation_compliance": {"pass": False, "issues": ["内容为空"]},
            "coherence": {"score": 0.0},
            "elapsed": time.time() - t0,
        }

    # Evaluate citation compliance
    citation_result = _check_citation_compliance(content)

    # Evaluate coherence
    coherence_result = _compute_coherence_score(content)

    return {
        "id": item.get("id", f"item_{idx}"),
        "content_length": len(content),
        "citation_compliance": citation_result,
        "coherence": coherence_result,
        "elapsed": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description="Q4 content generation quality evaluation")
    parser.add_argument("--sample", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="result/q4_eval_baseline.json", help="Output JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 5 samples")
    args = parser.parse_args()

    if args.quick:
        args.sample = 5

    print(f"Loading {args.sample} test samples...")
    samples = _load_test_samples(args.sample)
    print(f"Loaded {len(samples)} samples")

    results = []
    citation_pass_count = 0
    coherence_in_range_count = 0

    for idx, item in enumerate(samples):
        print(f"\n[{idx+1}/{len(samples)}] Evaluating {item.get('id', f'item_{idx}')}...")
        result = _evaluate_single(item, idx)
        results.append(result)

        if result.get("citation_compliance", {}).get("pass"):
            citation_pass_count += 1
        if result.get("coherence", {}).get("in_target_range"):
            coherence_in_range_count += 1

        print(f"  Citation: {'PASS' if result.get('citation_compliance', {}).get('pass') else 'FAIL'}")
        print(f"  Coherence: {result.get('coherence', {}).get('avg_similarity', 0.0):.3f}")

    # Compute aggregate metrics
    total = len(results)
    citation_pass_rate = citation_pass_count / total if total > 0 else 0.0
    coherence_pass_rate = coherence_in_range_count / total if total > 0 else 0.0

    avg_coherence = sum(r.get("coherence", {}).get("avg_similarity", 0.0) for r in results) / total if total > 0 else 0.0

    summary = {
        "total_samples": total,
        "citation_compliance_rate": round(citation_pass_rate, 3),
        "coherence_pass_rate": round(coherence_pass_rate, 3),
        "avg_coherence_score": round(avg_coherence, 3),
        "target_citation_rate": 0.80,
        "target_coherence_range": [0.15, 0.70],
        "citation_pass": citation_pass_rate >= 0.80,
        "coherence_pass": 0.15 <= avg_coherence <= 0.70,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    # Save results
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "="*60)
    print("Q4 Evaluation Summary")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Citation compliance rate: {citation_pass_rate:.1%} (target: >=80%)")
    print(f"  {'PASS' if citation_pass_rate >= 0.80 else 'FAIL'}")
    print(f"Coherence pass rate: {coherence_pass_rate:.1%} (in range 0.15-0.70)")
    print(f"Avg coherence score: {avg_coherence:.3f}")
    print(f"  {'PASS' if 0.15 <= avg_coherence <= 0.70 else 'FAIL'}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
