"""RAG pipeline metrics using real PDF papers.

Measures hit rate and noise rate with 27 Chinese academic papers.
Queries target specific sub-topics; gold papers identified by filename.

Usage:
  python scripts/eval_rag_pipeline.py
"""

import json, os, re, sys, tempfile, time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

os.environ["EMBED_API_STYLE"] = "ollama"
os.environ["EMBED_MODEL_NAME"] = "nomic-embed-text"

DATASET_DIR = _PROJ / ".Dataset"

# ── Main ────────────────────────────────────────────────────────────────────

def run():
    from agent_file_create.rag.kb import KnowledgeBase

    pdf_files = sorted(
        [str(p) for p in DATASET_DIR.glob("*.pdf") if not p.name.startswith(".")]
    )
    print(f"\n{'='*60}")
    print(f"  Planner Test: {len(pdf_files)} papers, 1 outline, 3 sections")
    print(f"{'='*60}")

    # Pre-flight: test embedding connectivity
    from agent_file_create.rag.embedder import embed_texts
    try:
        test_vecs = embed_texts(['测试文本'], timeout_s=30, max_batch=1)
        if test_vecs and len(test_vecs[0]) > 0:
            print(f"  Embed OK (dim={len(test_vecs[0])})")
        else:
            print(f"  Embed FAILED — empty vectors")
            return
    except Exception as e:
        print(f"  Embed FAILED — {e}")
        return
    print()

    # Use pre-existing KB (papers already embedded with nomic-embed-text)
    kb = KnowledgeBase()
    kb_name = "eval_papers_nomic"
    test_hits = kb.search(kb=kb_name, query='RAG', top_k=3)
    if not test_hits:
        print("  KB 'papers' empty — run full ingest first")
        return
    print(f"  KB 'papers': {len(test_hits)} hits — OK\n")

    # ── Planner test: section-type-aware planning ──────────────────────────
    print(f"\n{'='*75}")
    print(f"  PLANNER: section-type-aware knowledge planning")
    print(f"{'='*75}\n")

    # Simulate outlines with mixed section types
    test_outlines = [
        {
            "topic": "RAG系统优化",
            "outline": "# RAG系统优化\n"
                       "## RAG技术背景与相关工作\n"      # → review
                       "## 超参数优化实验与性能对比\n"    # → data
                       "## 未来研究方向与展望\n",          # → analysis
        },
    ]

    import agent_file_create.rag.planner as p

    from agent_file_create.document.content_generator import classify_section_type

    for ti, test in enumerate(test_outlines):
        outline = test["outline"]
        topic = test["topic"]
        print(f"  P{ti+1}: {topic}")
        print(f"    {'Section':<35} {'Type':>8} {'Points':>7} {'Hits':>5} {'压缩':>6}")
        print(f"    {'-'*61}")

        # Parse h2 sections
        h2s = [s for s in re.findall(r'^##\s+(.+)$', outline, re.MULTILINE)]

        for sec_title in h2s:
            sec_type = classify_section_type(sec_title)
            try:
                plan = p.plan_section_knowledge(
                    section_title=sec_title,
                    parent_title=topic,
                    user_prompt=topic,
                    kb=kb,
                    kb_name=kb_name,
                    max_points=3,
                )
                kps = len(plan.get("knowledge_points", []))
                mat_chars = len(plan.get("materials", ""))
                
                # Use hits count from optimized planner
                planner_hits = plan.get("hits_count", 0)
                
                # Debug: show what knowledge points were generated
                if plan.get("knowledge_points"):
                    print(f"    [DEBUG] {sec_title[:20]} KPs: {plan['knowledge_points'][:2]}")
                    # Direct search test (for comparison)
                    _th = kb.search(kb=kb_name, query=sec_title[:10], top_k=2)
                    print(f"    [DEBUG] direct search '{sec_title[:10]}': {len(_th)} hits (planner: {planner_hits} hits)")

                n_hits = planner_hits if planner_hits > 0 else len(kb.search(kb=kb_name, query=sec_title, top_k=5))

                ratio = ""
                if n_hits > 0 and mat_chars > 0:
                    ratio = f"{mat_chars} chars"

                print(f"    {sec_title[:35]:<35} {sec_type:>8} {kps:>7} {n_hits:>5} {ratio:>6}")
            except Exception as e:
                print(f"    {sec_title[:35]:<35} {'?':>8} {'err':>7} {'-':>5} {str(e)[:40]}")

        print()

    # KB retained for reuse


if __name__ == "__main__":
    run()
