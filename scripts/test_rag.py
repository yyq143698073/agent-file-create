#!/usr/bin/env python
"""Integration test — mirrors the web pipeline exactly, generating a ~2000-word RAG report.

Uses 3 PDFs from resource/, ingests into 'RAG知识' KB, runs the full
outline→planner→content→renumber chain just like the web DocumentAgent.

Usage:
    python scripts/test_rag.py

Output:
    result/rag_test/content.md + analysis summary

Checks:
    1. No repeated 【n】 in same paragraph (rule 9)
    2. Multiple unique citation IDs (rule 10)
    3. No placeholder text like （材料暂缺）
    4. References section present
"""

import os
import re
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

os.environ.setdefault("CONCURRENT_LLM_CALLS", "32")

RESOURCE_DIR = _project_root / "resource"
PDF_FILES = [
    "基于贝叶斯优化的RAG系统超参数调优_董千里.pdf",
    "基于动态规划的RAG语义感知分块方法_谢圣富.pdf",
    "检索增强生成推荐及其研究进展_吴国栋.pdf",
]
KB_NAME = "RAG知识"
PROMPT = "生成RAG优化报告"
TARGET_WORDS = 2000


def analyze(content: str) -> dict:
    numbered = re.findall(r'[【\[](\d+)[】\]]', content)
    ids = sorted(set(int(x) for x in numbered))
    dist = {i: numbered.count(str(i)) for i in ids} if ids else {}

    paras = [p.strip() for p in content.split('\n\n') if p.strip()]
    repeat_paras = 0
    for p in paras:
        markers = re.findall(r'[【\[](\d+)[】\]]', p)
        if len(set(markers)) < len(markers):
            repeat_paras += 1

    placeholders = re.findall(r'[（(]材料[暂缺不].*?[）)]', content)
    has_placeholders = len(placeholders) > 0

    has_refs = "## 参考文献" in content

    return {
        "chars": len(content),
        "h2": len(re.findall(r'^## ', content, re.MULTILINE)),
        "markers": len(numbered),
        "unique_ids": ids,
        "distribution": dist,
        "repeat_paragraphs": repeat_paras,
        "placeholders": placeholders[:5],
        "has_placeholders": has_placeholders,
        "has_refs": has_refs,
    }


def main():
    t0 = time.perf_counter()
    out_dir = Path("result/rag_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Verify files exist ──────────────────────────────────────────
    for fn in PDF_FILES:
        if not (RESOURCE_DIR / fn).exists():
            print(f"ERROR: {fn} not found in resource/")
            sys.exit(1)

    # ── 1. Ingest all 3 PDFs ───────────────────────────────────────────
    print(f"[1/5] Ingesting 3 PDFs into '{KB_NAME}'...")
    from agent_file_create.rag import get_kb
    kb = get_kb()
    kb.register_kb(KB_NAME)

    texts = {}
    for fn in PDF_FILES:
        fp = RESOURCE_DIR / fn
        from agent_file_create.preprocessor import read_text_file
        text = read_text_file(str(fp))
        texts[fn] = text
        r = kb.ingest_file(kb=KB_NAME, file_path=str(fp), title=fp.stem, doc_type="paper")
        print(f"  {fn}: {'OK' if r.get('ok') else r.get('error','?')} ({r.get('chunks','?')} chunks)")

    # ── 2. Build multimodal_results (same as web) ───────────────────────
    summary_text = "\n\n".join(t[:2000] for t in texts.values())
    multimodal_results = {
        "files": [
            {"title": Path(fn).stem, "filename": fn, "summary": texts[fn][:2000]}
            for fn in PDF_FILES
        ],
        "summaries": [summary_text[:5000]],
    }

    # ── 3. Generate outline ────────────────────────────────────────────
    print(f"[2/5] Generating outline ({TARGET_WORDS} words)...")
    from agent_file_create.document.outline_generator import generate_outline
    outline = generate_outline(
        multimodal_results=multimodal_results,
        user_prompt=PROMPT,
        target_words=TARGET_WORDS,
    )
    outline = re.sub(r'^\d+\.\s*', '', outline, flags=re.MULTILINE)
    (out_dir / "outline.md").write_text(outline, encoding="utf-8")
    h2s = len(re.findall(r'^## ', outline, re.MULTILINE))
    print(f"  {len(outline)} chars, {h2s} H2 sections")

    # ── 4. KB Planner (mirrors web _node_content exactly) ──────────────
    print(f"[3/5] Planning knowledge retrieval (mirrors web pipeline)...")
    from agent_file_create.rag.planner import (
        plan_all_sections, build_citation_map, _compress_hits_annotated,
        renumber_citations, format_citation_list,
    )
    from agent_file_create.document.content_generator import generate_full_content_parallel

    _plan = plan_all_sections(
        outline=outline, user_prompt=PROMPT,
        kb=kb, kb_name=KB_NAME, target_words=TARGET_WORDS,
    )

    # Build annotated materials + citation map (EXACTLY like document_agent.py lines 1090-1141)
    _all_cit_maps, _annotated_parts = {}, []
    for _sec_title, _sp in _plan.items():
        _raw = _sp.get("_raw_hits") or []
        if _raw:
            _annotated, _sec_cit_map = _compress_hits_annotated(
                _raw, _sp.get("knowledge_points", [_sec_title])[0],
                section_type=_sp.get("section_type", "review"),
            )
            _all_cit_maps[_sec_title] = {"citation_map": _sec_cit_map}
            if _annotated:
                _annotated_parts.append(f"## {_sec_title}\n{_annotated}")

    citation_map = build_citation_map(_all_cit_maps)
    enriched = ""
    if _annotated_parts:
        enriched = (enriched or "") + "\n\n---\n\n# 带编号引用的检索材料\n\n" + "\n\n".join(_annotated_parts)

    print(f"  {len(_plan)} sections, {len(citation_map)} unique citations, "
          f"{sum(len(p) for p in _annotated_parts)} annotated chars")

    # ── 5. Generate content (same call as web) ──────────────────────────
    print(f"[4/5] Generating content...")
    content = generate_full_content_parallel(
        outline=outline, multimodal_results=multimodal_results,
        user_prompt=PROMPT, task_id="rag_test",
        target_words=TARGET_WORDS, enriched_context=enriched,
    )

    # ── 6. Post-process (same as web) ──────────────────────────────────
    _orig_map = dict(citation_map)
    content, citation_map = renumber_citations(content, citation_map)
    _ref_map = citation_map if citation_map else _orig_map
    refs = format_citation_list(_ref_map) if _ref_map else ""
    if not refs:
        ids = sorted(set(int(m) for m in re.findall(r'[【\[](\d+)[】\]]', content)))
        if ids:
            refs = "\n".join(f"【{i}】 来源信息见正文引用标注" for i in ids)
    if refs and "## 参考文献" not in content:
        content = content.rstrip() + "\n\n---\n\n## 参考文献\n\n" + refs

    (out_dir / "content.md").write_text(content, encoding="utf-8")

    # ── Analyze ─────────────────────────────────────────────────────────
    stats = analyze(content)
    t1 = time.perf_counter()

    print(f"\n[5/5] Results ({t1 - t0:.1f}s):")
    print(f"  Chars: {stats['chars']}  H2 sections: {stats['h2']}")
    print(f"  Markers: {stats['markers']}  Unique IDs: {stats['unique_ids']}")
    print(f"  Distribution: {stats['distribution']}")
    print(f"  Repeat paragraphs: {stats['repeat_paragraphs']}")
    print(f"  Placeholders: {stats['placeholders']}")
    print(f"  References: {'FOUND' if stats['has_refs'] else 'MISSING'}")

    # ── Verdict ─────────────────────────────────────────────────────────
    issues = []
    if stats['repeat_paragraphs'] > 2:
        issues.append(f"{stats['repeat_paragraphs']} paragraphs have repeated same 【n】")
    if len(stats['unique_ids']) <= 1 and stats['markers'] >= 5:
        issues.append("Only 1 unique citation ID (all 【1】)")
    if stats['has_placeholders']:
        issues.append(f"Found placeholders: {stats['placeholders']}")
    if not stats['has_refs']:
        issues.append("No references section")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for i in issues:
            print(f"    - {i}")
    else:
        print(f"\n  ALL CHECKS PASSED")

    print(f"  Content: {out_dir / 'content.md'}")


if __name__ == "__main__":
    main()
