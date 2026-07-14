# -*- coding: utf-8 -*-
"""
检索质量测试: 验证查询扩展 / 质量检查 / 概念提取等纯函数逻辑
运行: python tests/test_retrieval_quality.py

覆盖(P0):
  - _generate_expanded_queries(): 查询扩展
  - _quality_ok(): 检索质量检查
  - _extract_terms(): 关键词提取
  - _extract_concepts_from_title(): 标题概念提取
  - _extract_year(): 年份提取
  - _get_context_budget(): 上下文预算
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_file_create.rag.planner import (
    _generate_expanded_queries,
    _quality_ok,
    _extract_terms,
    _extract_concepts_from_title,
    _extract_year,
    _get_context_budget,
)
from agent_file_create.rag.store import Hit
from tests.test_utils import print_section_header


def test_query_expansion():
    """Test _generate_expanded_queries produces diverse, relevant queries."""
    cases = [
        {
            "name": "Technical term",
            "input": "RAG系统的检索质量优化方法",
            "min_expected": 3,
            "must_contain": ["RAG"],
        },
        {
            "name": "English acronym",
            "input": "DPR编码器在NQ数据集上的表现",
            "min_expected": 3,
            "must_contain": ["DPR"],
        },
        {
            "name": "Short query",
            "input": "知识图谱",
            "min_expected": 1,
        },
        {
            "name": "Empty query",
            "input": "",
            "min_expected": 0,
        },
    ]

    passed = 0
    for case in cases:
        queries = _generate_expanded_queries(case["input"])
        ok = len(queries) >= case["min_expected"]
        if case.get("must_contain"):
            ok = ok and all(
                any(m in q for q in queries) for m in case["must_contain"]
            )
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {case['name']}: {len(queries)} queries")
        if not ok:
            print(f"           expected >= {case['min_expected']}, got {len(queries)}")
        if queries:
            print(f"           first 3: {queries[:3]}")
    return passed, len(cases)


def test_quality_ok():
    """Test _quality_ok correctly identifies good/poor retrieval."""
    def make_hit(doc_id, score):
        return Hit(kb='test', doc_id=doc_id, chunk_id=f'{doc_id}_c1',
                   chunk_index=0, section_path='', content='test',
                   score=score, meta={})

    cases = [
        # (hits, min_score, min_unique, min_hits, expected)
        ("3 good hits", [make_hit("a", 0.9), make_hit("b", 0.8), make_hit("c", 0.75)],
         None, None, None, True),
        ("1 good + 4 bad (permissive defaults)", [make_hit("a", 0.95)] + [make_hit(f"b{i}", 0.1) for i in range(4)],
         None, None, None, True),  # only 1 good top-3
        ("Empty", [], None, None, None, False),
        ("Low batch quality", [make_hit("a", 0.2), make_hit("b", 0.15)], 0.5, 1, 2, False),
    ]

    passed = 0
    for name, hits, min_s, min_u, min_h, expected in cases:
        kwargs = {}
        if min_s is not None: kwargs["min_score"] = min_s
        if min_u is not None: kwargs["min_unique_docs"] = min_u
        if min_h is not None: kwargs["min_hits"] = min_h
        result = _quality_ok(hits, **kwargs)
        ok = result == expected
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: got={result}, expected={expected}")
    return passed, len(cases)


def test_extract_terms():
    """Test _extract_terms extracts meaningful Chinese/English terms."""
    cases = [
        ("Technical phrase", "RAG检索增强生成技术", 3),
        ("English only", "DPR BART T5", 3),
        ("Empty", "", 0),
        ("Mixed", "2024年新能源汽车销量分析", 3),
    ]
    passed = 0
    for name, text, min_terms in cases:
        terms = _extract_terms(text)
        ok = len(terms) >= min_terms
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {len(terms)} terms {terms[:4]}")
    return passed, len(cases)


def test_concepts_from_title():
    """Test _extract_concepts_from_title extracts core concepts."""
    cases = [
        ("RAG技术背景与相关工作", ["RAG", "RAG技术", "技术背景"]),
        ("实验结果分析", ["实验", "结果", "分析"]),
        ("", []),
    ]
    passed = 0
    for name, title, expected_kw in cases:
        concepts = _extract_concepts_from_title(title)
        ok = expected_kw in concepts if expected_kw else len(concepts) == 0
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {concepts[:5]}")
    return passed, len(cases)


def test_extract_year():
    """Test _extract_year extracts publication year from hit metadata."""

    class MockHit:
        def __init__(self, doc_id="", content="", meta=None):
            self.doc_id = doc_id
            self.content = content
            self.meta = meta or {}
            self.chunk_id = ""
            self.section_path = ""
            self.score = 0.0

    cases = [
        ("From meta", MockHit(meta={"year": 2024}), True),
        ("From doc_id", MockHit(doc_id="paper_2023.pdf"), True),
        ("From content", MockHit(content="该研究发表于2022年"), True),
        ("No info", MockHit(), False),
    ]
    passed = 0
    for name, hit, expect_found in cases:
        year = _extract_year(hit)
        ok = bool(year) == expect_found
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: year='{year}'")
    return passed, len(cases)


def test_context_budget():
    """Test _get_context_budget computes reasonable budgets."""
    checks = [
        ("data > review", _get_context_budget("data", 8000) > _get_context_budget("review", 0)),
        ("data scales with target", _get_context_budget("data", 8000) > _get_context_budget("data", 0)),
        ("min 400 chars", all(_get_context_budget(t, 0) >= 400 for t in ["data", "analysis", "review"])),
    ]
    passed = 0
    for name, ok in checks:
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    return passed, len(checks)


def run():
    print_section_header("RETRIEVAL QUALITY: Query Expansion")
    r1, t1 = test_query_expansion()

    print_section_header("RETRIEVAL QUALITY: Quality Check (_quality_ok)")
    r2, t2 = test_quality_ok()

    print_section_header("RETRIEVAL QUALITY: Term Extraction (_extract_terms)")
    r3, t3 = test_extract_terms()

    print_section_header("RETRIEVAL QUALITY: Concepts from Title")
    r4, t4 = test_concepts_from_title()

    print_section_header("RETRIEVAL QUALITY: Year Extraction (_extract_year)")
    r5, t5 = test_extract_year()

    print_section_header("RETRIEVAL QUALITY: Context Budget")
    r6, t6 = test_context_budget()

    total = t1 + t2 + t3 + t4 + t5 + t6
    passed = r1 + r2 + r3 + r4 + r5 + r6

    print_section_header("SUMMARY")
    print(f"  Query Expansion:    {r1}/{t1}")
    print(f"  Quality Check:      {r2}/{t2}")
    print(f"  Term Extraction:    {r3}/{t3}")
    print(f"  Concepts Extraction:{r4}/{t4}")
    print(f"  Year Extraction:    {r5}/{t5}")
    print(f"  Context Budget:     {r6}/{t6}")
    print(f"  Total:              {passed}/{total}")
    print(f"  Score: {passed}/{total} ({passed/total*100:.0f}%)")
    print_section_header("Done.")


if __name__ == "__main__":
    run()
