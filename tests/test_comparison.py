# -*- coding: utf-8 -*-
"""
对比测试: 量化 Critic + 覆盖度 + 幻觉硬化的提升效果
运行: MODEL=qwen3.5:9b python tests/test_comparison.py

优化(P4): 共享config/metrics迁移到test_utils.py
"""
import asyncio, json, os, re, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_file_create.document._reviewer import (
    extract_facts_from_materials, cross_check_facts, patch_unverified_claims,
    detect_cross_document_conflicts,
)
from agent_file_create.rag.planner import (
    _get_context_budget, _compress_hits_annotated,
    verify_citations, renumber_citations,
)
from agent_file_create.document.content_generator import _compute_coverage_map
from agent_file_create.rag.store import Hit
from tests.test_utils import count_numbers, count_citations, count_placeholder, print_section_header

# ── Config (use env vars) ──
STYLE   = os.getenv("STYLE",   "ollama")
MODEL   = os.getenv("MODEL",   "qwen3.5:9b")
ENDPOINT = os.getenv("ENDPOINT", "http://localhost:11434")
KEY     = os.getenv("KEY",     "")


# ── Test Cases ──
TEST_CASES = [
    {
        "name": "Case 1: Data errors (numbers)",
        "materials": (
            "2024年新能源车全球销量达1200万辆，同比增长35%。"
            "比亚迪市场份额32%，特斯拉18%，蔚来8%。"
            "纯电车型占比70%，插混25%，氢燃料5%。"
        ),
        "content": (
            "2024年新能源车全球销量约为800万辆，同比增长28%。"
            "比亚迪占据最大市场份额约为45%，特斯拉居第二。"
            "纯电车型占据主导地位约80%，插混约15%。"
            "氢燃料电池技术已被市场淘汰。"
        ),
        "expected_issues": 4,
    },
    {
        "name": "Case 2: Missing information",
        "materials": (
            "Chatchat项目支持以下向量数据库：FAISS（本地文件）、"
            "Milvus（分布式）、PostgreSQL+pgvector、Elasticsearch、ChromaDB。"
            "默认使用FAISS，配置在kb_settings.yaml的kbs_config中。"
            "同时支持中文优化的ChineseRecursiveTextSplitter，chunk_size默认750。"
        ),
        "content": (
            "Chatchat项目支持FAISS向量数据库。"
            "配置在kb_settings.yaml中。"
        ),
        "expected_issues": 1,
    },
    {
        "name": "Case 3: Cross-document contradiction",
        "materials": (
            "[paper_a.pdf] 实验显示RAG方法在准确率上达到95.3%，召回率89.1%。\n"
            "[paper_b.pdf] 同一数据集上RAG方法准确率为92.1%，召回率87.5%。"
        ),
        "content": (
            "RAG方法在该数据集上的准确率约为93%，性能表现优异。"
        ),
        "expected_issues": 1,
    },
]


async def run_comparison():
    print(f"Regex-layer comparison (no LLM required)\n")
    print(f"Model config: {MODEL} @ {ENDPOINT}\n")

    results = []
    total_before_issues = 0
    total_after_issues = 0

    for case in TEST_CASES:
        name = case["name"]
        materials = case["materials"]
        content = case["content"]
        print_section_header(name)

        # ── Phase A: WITHOUT hardening ──
        facts = extract_facts_from_materials(materials)
        raw_issues = cross_check_facts(content, facts)
        before_issue_count = len(raw_issues)
        total_before_issues += before_issue_count

        print(f"\n  [Without Critic]")
        print(f"  Material numbers: {sorted(facts.get('numbers', set()))[:10]}")
        print(f"  Content numbers:  {count_numbers(content)}")
        print(f"  Regex issues:     {before_issue_count}")
        for ri in raw_issues[:6]:
            print(f"    - {ri}")

        # ── Phase B: WITH hardening ──
        patched, patches = patch_unverified_claims(content, facts)
        after_issue_count = count_placeholder(patched)
        total_after_issues += after_issue_count

        print(f"\n  [With Critic]")
        print(f"  Auto-patches:     {len(patches)}")
        for p in patches[:6]:
            print(f"    - {p}")
        print(f"  [待核实数据]:     {count_placeholder(patched)}")

        # ── Phase C: Coverage check ──
        kps = []
        for m in re.finditer(r'(\w{2,}(?:技术|方法|项目|系统|平台|数据库|车型|电池))', materials):
            kps.append(m.group())
        kps = list(set(kps))[:5]
        if not kps:
            kps = [materials[:20]]
        coverage = _compute_coverage_map(kps, materials)
        covered = sum(1 for _, s in coverage if s in ("充足", "有限"))
        uncovered = sum(1 for _, s in coverage if s == "无")

        print(f"\n  [Coverage Map (jieba-based)]")
        print(f"  Knowledge points: {len(kps)}")
        print(f"  Covered: {covered}, Uncovered: {uncovered}")
        for kp, status in coverage:
            marker = "OK" if status != "无" else "!!"
            print(f"    {marker} [{status}] {kp}")

        # ── Phase D: Citation annotation demo ──
        hits_for_cite = [
            Hit(kb='t', doc_id='source_1.pdf', chunk_id=f'c{i}', chunk_index=i,
                section_path='', content=sent[:200],
                score=0.9, meta={'year': 2024 if i % 2 == 0 else 2023})
            for i, sent in enumerate(materials.split('。')[:5]) if sent.strip()
        ]
        if hits_for_cite:
            annotated, cit_map = _compress_hits_annotated(
                hits_for_cite, 'test', section_type='review',
            )
            cite_count = count_citations(annotated)
            print(f"\n  [Citation Annotation]")
            print(f"  Sources annotated: {len(cit_map)}")
            print(f"  Citation markers:  {cite_count}")

        # ── Phase E: Cross-doc contradiction ──
        if "paper_" in materials.lower() or "pdf" in materials:
            doc_hits = []
            for i, seg in enumerate(materials.split('\n')):
                seg = seg.strip()
                if not seg:
                    continue
                doc_name = f"doc_{i}.pdf"
                m = re.match(r'\[(\S+)\]', seg)
                if m:
                    doc_name = m.group(1)
                    seg = seg[len(m.group(0)):].strip()
                doc_hits.append(Hit(
                    kb='t', doc_id=doc_name, chunk_id=f'c{i}', chunk_index=i,
                    section_path='', content=seg[:300], score=0.9, meta={},
                ))
            if len(doc_hits) >= 2:
                conflicts = detect_cross_document_conflicts(doc_hits)
                print(f"\n  [Cross-Doc Conflicts]")
                print(f"  Documents checked: {len(doc_hits)}")
                print(f"  Conflicts found:   {len(conflicts)}")
                for c in conflicts[:3]:
                    claims_str = ", ".join(f"{doc}: {val}" for val, doc in c['claims'].items())
                    print(f"    - {c['metric']}: {claims_str}")

        results.append({
            "name": name,
            "before_issues": before_issue_count,
            "after_fixes": len(patches),
            "after_placeholders": after_issue_count,
            "covered_kps": covered,
            "uncovered_kps": uncovered,
        })
        print()

    # ── Summary ──
    print_section_header("COMPARISON SUMMARY")
    print(f"  {'Case':<40} {'Before':>7} {'Fixed':>7} {'Coverage':>10}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*10}")
    for r in results:
        cov_str = f"{r['covered_kps']}/{r['covered_kps']+r['uncovered_kps']}"
        print(f"  {r['name']:<40} {r['before_issues']:>7} {r['after_fixes']:>7} {cov_str:>10}")

    print(f"\n  Total regex issues before Critic: {total_before_issues}")
    print(f"  Total auto-fixes applied:         {sum(r['after_fixes'] for r in results)}")
    print(f"  Placeholders after hardening:     {total_after_issues}")
    if total_before_issues > 0:
        reduction = (1 - total_after_issues / max(total_before_issues, 1)) * 100
        print(f"  Hallucination reduction: {reduction:.0f}%")
    print(f"\n  {'='*60}")
    print(f"  Done.")

if __name__ == "__main__":
    asyncio.run(run_comparison())
