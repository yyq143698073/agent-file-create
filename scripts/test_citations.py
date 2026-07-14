#!/usr/bin/env python
"""End-to-end citation diversity test — generates a report with multi-source KB.

Verifies: when multiple files are in the KB, the LLM uses distinct 【n】
numbers for different sources instead of all 【1】.

Usage:
    python scripts/test_citations.py --kb RAG知识 --words 1500
    python scripts/test_citations.py --kb 一个KB --file test.pdf
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

os.environ.setdefault("CONCURRENT_LLM_CALLS", "32")


def count_citations(content: str) -> dict:
    """Analyze citation diversity in generated content."""
    numbered = re.findall(r'[【\[](\d+)[】\]]', content)
    natural = re.findall(r'（据.+?）', content)
    ids = [int(x) for x in numbered]

    return {
        "total_numbered": len(numbered),
        "unique_ids": sorted(set(ids)),
        "id_distribution": {i: ids.count(i) for i in sorted(set(ids))},
        "natural_citations": len(natural),
        "single_id_only": len(set(ids)) == 1,
    }


def main():
    ap = argparse.ArgumentParser(description="E2E citation diversity test")
    ap.add_argument("--kb", default="test_cite_kb", help="KB name to use")
    ap.add_argument("--file", help="PDF to ingest (required if KB is empty)")
    ap.add_argument("--words", type=int, default=1500)
    ap.add_argument("--prompt", default="综合分析这些文献的研究方法和主要发现，撰写一份综述报告")
    args = ap.parse_args()

    # ── 0. Ensure KB has multiple files ─────────────────────────────
    from agent_file_create.rag import get_kb
    kb = get_kb()
    kb.register_kb(args.kb)
    docs = kb.list_docs(kb=args.kb)
    if len(docs) < 2:
        if not args.file:
            print(f"ERROR: KB '{args.kb}' has only {len(docs)} file(s). Need >= 2.")
            print(f"       Provide --file <pdf> to ingest, or use a KB with multiple files.")
            print(f"       Available KBs: {kb.list_kb()}")
            sys.exit(1)
        print(f"Ingesting {args.file} into '{args.kb}'...")
        fp = Path(args.file)
        if not fp.exists():
            print(f"ERROR: {args.file} not found")
            sys.exit(1)
        r = kb.ingest_file(kb=args.kb, file_path=str(fp), title=fp.stem)
        print(f"  {'OK' if r.get('ok') else 'FAILED: ' + r.get('error', '')}")
        docs = kb.list_docs(kb=args.kb)

    print(f"\nKB '{args.kb}': {len(docs)} files")
    for d in docs[:10]:
        print(f"  - {d.get('title', '?')}")

    # ── 1. Build a dummy multimodal digest from KB docs ─────────────
    # We need at least some text to generate an outline from
    sample_text = ""
    for d in docs[:5]:
        t = kb.get_doc_text(kb=args.kb, doc_id=d.get("doc_id", ""))
        if t:
            sample_text += t[:600] + "\n\n"
    if not sample_text:
        print("ERROR: no text extracted from KB")
        sys.exit(1)

    multimodal_results = {
        "files": [{"title": d.get("title", "?"), "filename": d.get("title", "?"),
                    "summary": kb.get_doc_text(kb=args.kb, doc_id=d.get("doc_id", ""))[:800]}
                  for d in docs[:8]],
        "summaries": [sample_text[:3000]],
    }

    # ── 2. Generate outline ─────────────────────────────────────────
    print(f"\n[1/3] Generating outline...")
    t0 = time.perf_counter()
    from agent_file_create.document.outline_generator import generate_outline
    outline = generate_outline(
        multimodal_results=multimodal_results,
        user_prompt=args.prompt,
        target_words=args.words,
    )
    outline = re.sub(r'^\d+\.\s*', '', outline, flags=re.MULTILINE)
    h2_count = len(re.findall(r'^##\s+', outline, re.MULTILINE))
    print(f"      {len(outline)} chars, {h2_count} H2 sections")

    # ── 3. KB Planner + Content ─────────────────────────────────────
    print(f"[2/3] Planning + generating content...")
    from agent_file_create.rag.planner import (
        plan_all_sections, build_citation_map, _compress_hits_annotated,
        renumber_citations,
    )
    from agent_file_create.document.content_generator import generate_full_content_parallel

    _plan = plan_all_sections(outline=outline, user_prompt=args.prompt,
                              kb=kb, kb_name=args.kb, target_words=args.words)

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
        enriched = "\n\n---\n\n# 带编号引用的检索材料\n\n" + "\n\n".join(_annotated_parts)

    print(f"      {len(_plan)} sections, {len(citation_map)} unique citations, "
          f"{sum(len(p) for p in _annotated_parts)} annotated chars")

    content = generate_full_content_parallel(
        outline=outline, multimodal_results=multimodal_results,
        user_prompt=args.prompt, task_id="cite_test",
        target_words=args.words, enriched_context=enriched,
    )

    # ── Renumber + analyze ───────────────────────────────────────────
    pre_stats = count_citations(content)
    if citation_map:
        content, citation_map = renumber_citations(content, citation_map)
    post_stats = count_citations(content)

    # ── 4. Report ────────────────────────────────────────────────────
    t1 = time.perf_counter()
    out_dir = Path("result/cite_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "content.md").write_text(content, encoding="utf-8")

    print(f"\n[3/3] Results ({t1 - t0:.1f}s total):")
    print(f"  Pre-renumber:  {pre_stats['total_numbered']} markers, "
          f"unique={pre_stats['unique_ids']}")
    print(f"  Post-renumber: {post_stats['total_numbered']} markers, "
          f"unique={post_stats['unique_ids']}")
    print(f"  Natural citations: {post_stats['natural_citations']}")
    print(f"  Distribution: {post_stats['id_distribution']}")
    print(f"  Content: {out_dir / 'content.md'} ({len(content)} chars)")

    # ── Verdict ──────────────────────────────────────────────────────
    if post_stats['single_id_only']:
        print(f"\n  VERDICT: FAIL — all citations use same number")
    elif post_stats['total_numbered'] < len(post_stats['unique_ids']) * 3:
        print(f"\n  VERDICT: OK — citations are diverse but sparse")
    else:
        print(f"\n  VERDICT: PASS — citations are diverse and frequent")


if __name__ == "__main__":
    main()
