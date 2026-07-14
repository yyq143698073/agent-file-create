#!/usr/bin/env python
"""Quick document generation with RAG — bypasses agent graph for fast testing.

Usage:
    python scripts/quick_gen.py <file> [prompt]
    python scripts/quick_gen.py test.pdf "总结这篇论文" --kb my_kb --words 1500
    python scripts/quick_gen.py paper.pdf --kb 一个KB

Steps:
    1. Extract text from file
    2. Ingest into KB (if --kb provided) or use file content directly
    3. Generate outline
    4. Use KB planner + parallel content generation
    5. Write output

Output: result/quick_test/{outline.md, content.md}
"""

import argparse
import os
import re
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    ap = argparse.ArgumentParser(description="Quick RAG report generation (~1500 words)")
    ap.add_argument("file", help="Input file path (PDF/Word/PPT/Excel/txt)")
    ap.add_argument("prompt", nargs="?", default="总结并分析这份文档",
                    help="Generation prompt")
    ap.add_argument("--words", type=int, default=1500, help="Target word count")
    ap.add_argument("--kb", default="quick_test_kb", help="KB name for ingestion + retrieval")
    ap.add_argument("--out", default="result/quick_test", help="Output directory")
    args = ap.parse_args()

    fp = Path(args.file)
    if not fp.exists():
        print(f"ERROR: file not found: {args.file}")
        sys.exit(1)

    # Set high concurrency for speed
    os.environ.setdefault("CONCURRENT_LLM_CALLS", "32")

    t0 = time.perf_counter()
    task_id = uuid.uuid4().hex[:8]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract ───────────────────────────────────────────────
    print(f"[1/5] Extracting {fp.name} ...")
    from agent_file_create.preprocessor import read_text_file
    text = read_text_file(str(fp))
    if not text.strip():
        print("ERROR: no text extracted")
        sys.exit(1)
    print(f"      {len(text)} chars")

    # ── Step 2: Ingest into KB ─────────────────────────────────────────
    print(f"[2/5] Ingesting into KB '{args.kb}' ...")
    from agent_file_create.rag import get_kb
    kb = get_kb()

    # Register KB if new
    kb.register_kb(args.kb)

    result = kb.ingest_file(kb=args.kb, file_path=str(fp), title=fp.stem, doc_type="paper")
    if not result.get("ok"):
        print(f"WARN: ingest skipped — {result.get('error', 'unknown')} — using direct text")
    else:
        print(f"      {result.get('chunks', '?')} chunks ingested")

    # ── Step 3: Outline ────────────────────────────────────────────────
    print(f"[3/5] Generating outline ({args.words} words) ...")
    multimodal_results = {
        "files": [{"title": fp.stem, "filename": fp.name, "summary": text[:3000]}],
        "summaries": [text[:3000]],
    }
    from agent_file_create.document.outline_generator import generate_outline
    outline = generate_outline(
        multimodal_results=multimodal_results,
        user_prompt=args.prompt,
        target_words=args.words,
    )
    outline = re.sub(r'^\d+\.\s*', '', outline, flags=re.MULTILINE)
    (out_dir / "outline.md").write_text(outline, encoding="utf-8")
    print(f"      {len(outline)} chars, ~{outline.count('#')} sections")

    # ── Step 4: KB Planner (knowledge retrieval per section) ────────────
    print(f"[4/5] Planning knowledge retrieval ...")
    from agent_file_create.rag.planner import plan_all_sections, build_citation_map, format_citation_list
    from agent_file_create.rag.planner import renumber_citations

    _plan, enriched, citation_map = {}, "", {}
    _annotated_parts = []
    try:
        _plan = plan_all_sections(
            outline=outline,
            user_prompt=args.prompt,
            kb=kb,
            kb_name=args.kb,
            target_words=args.words,
        )
        enriched = ""  # annotated materials appended below
        # Build annotated materials + citation map — same as document_agent flow
        _all_cit_maps = {}
        for _sec_title, _sp in _plan.items():
            _raw = _sp.get("_raw_hits") or []
            if _raw:
                from agent_file_create.rag.planner import _compress_hits_annotated
                _annotated, _sec_cit_map = _compress_hits_annotated(
                    _raw,
                    _sp.get("knowledge_points", [_sec_title])[0],
                    section_type=_sp.get("section_type", "review"),
                )
                _all_cit_maps[_sec_title] = {"citation_map": _sec_cit_map}
                if _annotated:
                    _annotated_parts.append(f"## {_sec_title}\n{_annotated}")
        citation_map = build_citation_map(_all_cit_maps)
        # Critical: append annotated materials to enriched_context so LLM sees 【1】【2】 markers
        if _annotated_parts:
            enriched = (enriched or "") + "\n\n---\n\n# 带编号引用的检索材料\n\n" + "\n\n".join(_annotated_parts)
        print(f"      {len(_plan)} sections, {len(citation_map)} unique citations, {sum(len(p) for p in _annotated_parts)} annotated chars")
    except Exception as e:
        print(f"      planner skipped: {e}")

    # ── Step 5: Content Generation ──────────────────────────────────────
    print(f"[5/5] Generating content ...")
    from agent_file_create.document.content_generator import generate_full_content_parallel

    content = generate_full_content_parallel(
        outline=outline,
        multimodal_results=multimodal_results,
        user_prompt=args.prompt,
        task_id=task_id,
        target_words=args.words,
        enriched_context=enriched,
    )

    # ── Post-processing ─────────────────────────────────────────────────
    _ref_map = dict(citation_map)  # start with original
    if citation_map:
        print(f"      pre-renumber: {len(citation_map)} entries, {len(re.findall(r'【', content))} markers")
        _orig_map = dict(citation_map)
        content, citation_map = renumber_citations(content, citation_map)
        _ref_map = citation_map if citation_map else _orig_map
        print(f"      post-renumber: {len(_ref_map)} entries")
    if _ref_map:
        refs = format_citation_list(_ref_map)
        if refs:
            content += "\n\n---\n\n## 参考文献\n\n" + refs

    (out_dir / "content.md").write_text(content, encoding="utf-8")

    t1 = time.perf_counter()
    print(f"\nDone in {t1 - t0:.1f}s")
    print(f"  Outline: {out_dir / 'outline.md'}")
    print(f"  Content: {out_dir / 'content.md'}  ({len(content)} chars)")


if __name__ == "__main__":
    main()
