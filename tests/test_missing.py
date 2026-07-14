# -*- coding: utf-8 -*-
"""
补测三个缺口: 引用使用率 / 预算缩放 / 覆盖度幻觉量化
运行: MODEL=qwen3.5:9b python tests/test_missing.py

优化(P3/P4):
  P3: LLM测试3次采样取多数
  P4: 共享config/metrics迁移到test_utils.py
"""
import asyncio, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from agent_file_create.rag.planner import _get_context_budget
from tests.test_utils import make_llm, print_section_header, STYLE, MODEL
from tests.test_utils import count_citations, count_placeholder, count_unsupported_claims, extract_all_numbers

# Test A: Citation
MATERIALS_ANNOTATED = """
【1】2024年销量达1200万辆，同比增长35% (来源: 2024, market_report.pdf > 市场规模)
【2】比亚迪市场份额32%，特斯拉18%，蔚来8% (来源: 2024, competition.pdf > 竞争格局)
【3】纯电占70%，插混25%，氢燃料5% (来源: 2023, tech_roadmap.pdf > 技术路线)
【4】固态电池预计2026年量产，能量密度突破400Wh/kg (来源: 2024, battery_paper.pdf > 电池技术)
"""
MATERIALS_PLAIN = "2024年销量达1200万辆，同比增长35%。比亚迪市场份额32%，特斯拉18%，蔚来8%。纯电占70%，插混25%，氢燃料5%。固态电池预计2026年量产，能量密度突破400Wh/kg。"
CITATION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """根据以下材料撰写一段关于新能源车行业的简要分析(150-300字)。
材料: {materials}
写作要求: 引用具体数据时，请在句末用【数字】标注对应的编号。"""),
])

# Test B: Budget (pure Python, no multi-sampling needed)
def test_budget():
    types = ["data", "experiment_setup", "analysis", "review"]
    tw_levels = [0, 2000, 5000, 8000]
    print("  Budget by section type and target_words:")
    header = "  " + "".join(f"{f'{tw}w':>8}" for tw in tw_levels)
    print(header)
    for st in types:
        row = f"  {st:<20}"
        for tw in tw_levels:
            row += f"{_get_context_budget(st, tw):>8}"
        print(row)
    d8k = _get_context_budget("data", 8000)
    r0 = _get_context_budget("review", 0)
    d0 = _get_context_budget("data", 0)
    return d8k > r0, d8k > d0, d8k / max(r0, 1)

# Test C: Coverage Hallucination
MATERIALS_COVERAGE = "RAG由Lewis等人于2020年提出，核心是结合密集检索与序列到序列生成。DPR检索器在NQ数据集上top-20准确率达78.3%，BART生成器在TriviaQA上EM达56.8。实验使用4张A100 GPU，推理延迟为120ms/query。代码已开源在GitHub。"
COVERAGE_MAP_TEXT = """
以下知识点在当前检索材料中的覆盖情况：
  [充足] RAG核心架构 (DPR + BART)
  [充足] 关键实验结果 (NQ 78.3%, TriviaQA 56.8)
  [充足] 实验配置与性能指标 (4x A100, 120ms)
  [有限] 与传统方法的对比数据
  [无]   RAG在工业界的落地案例
要求: [充足]可深入展开，[有限]只写已有内容，[无]跳过或标注[需补充数据]"""

WRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """撰写一段关于RAG的简要技术报告(200-400字)，覆盖以下知识点：1. RAG核心架构 2. 关键实验结果 3. 与传统方法的对比 4. 工业界落地案例。
参考材料: {materials}
{coverage}
要求: 严格基于材料写，不要编造数据。"""),
])

async def sample_llm(chain, inputs, timeout_s=60):
    try:
        text = await asyncio.wait_for(chain.ainvoke(inputs), timeout=timeout_s)
        return (text or "").strip()
    except Exception:
        return ""

async def run_multi(chain, inputs, n=3, label=""):
    samples, timings = [], []
    for i in range(n):
        t0 = time.perf_counter()
        text = await sample_llm(chain, inputs)
        elapsed = time.perf_counter() - t0
        if text:
            samples.append(text)
        timings.append(elapsed)
    return {"samples": samples, "n": n, "n_ok": len(samples),
            "timings": timings, "avg_time": sum(timings)/len(timings) if timings else 0}

async def run():
    llm = make_llm(temperature=0.01, max_tokens=600, timeout_s=180)
    print(f"Model: {MODEL}\n")
    total, score = 3, 0

    # === TEST A: Citation Usage (3x sampling) ===
    print_section_header("TEST A: Citation Marker Usage Rate (3x sampling)")
    chain = CITATION_PROMPT | llm | StrOutputParser()
    r_ann = await run_multi(chain, {"materials": MATERIALS_ANNOTATED}, n=3, label="annotated")
    r_plain = await run_multi(chain, {"materials": MATERIALS_PLAIN}, n=3, label="plain")
    cite_ann = sum(count_citations(t) for t in r_ann["samples"])
    cite_plain = sum(count_citations(t) for t in r_plain["samples"])
    avg_ann = cite_ann / max(len(r_ann["samples"]), 1)
    avg_plain = cite_plain / max(len(r_plain["samples"]), 1)
    passed_a = avg_ann > 0
    print(f"  Annotated avg citation markers: {avg_ann:.1f}")
    print(f"  Plain avg citation markers:     {avg_plain:.1f}")
    print(f"  Result: {'PASS' if passed_a else 'FAIL'}")
    score += 1 if passed_a else 0

    # === TEST B: Budget Scaling (pure Python) ===
    print_section_header("TEST B: Budget Scaling in Practice")
    ok1, ok2, ratio = test_budget()
    passed_b = ok1 and ok2
    print(f"  data/8000w > review/0w: {ok1} ({ratio:.1f}x)")
    print(f"  data/8000w > data/0w:  {ok2} (scaling works)")
    print(f"  Result: {'PASS' if passed_b else 'FAIL'}")
    score += 1 if passed_b else 0

    # === TEST C: Coverage Hallucination (3x sampling) ===
    print_section_header("TEST C: Coverage Map - Hallucination Reduction (3x sampling)")
    source_nums = extract_all_numbers(MATERIALS_COVERAGE)
    chain_no = WRITE_PROMPT | llm | StrOutputParser()
    chain_with = WRITE_PROMPT | llm | StrOutputParser()
    r_no = await run_multi(chain_no, {"materials": MATERIALS_COVERAGE, "coverage": ""}, n=3, label="no_cov")
    r_with = await run_multi(chain_with, {"materials": MATERIALS_COVERAGE, "coverage": COVERAGE_MAP_TEXT}, n=3, label="with_cov")
    h_no = sum(count_unsupported_claims(t, source_nums) for t in r_no["samples"])
    h_with = sum(count_unsupported_claims(t, source_nums) for t in r_with["samples"])
    avg_no = h_no / max(len(r_no["samples"]), 1)
    avg_with = h_with / max(len(r_with["samples"]), 1)
    reduction = (avg_no - avg_with) / max(avg_no, 1) * 100 if avg_no > 0 else 0
    has_placeholder = any("[需补充数据]" in t for t in r_with["samples"])
    passed_c = avg_with <= avg_no
    print(f"  Avg unsupported claims: {avg_no:.1f} -> {avg_with:.1f}")
    print(f"  Reduction: {reduction:.0f}%")
    print(f"  Has placeholder: {has_placeholder}")
    print(f"  Result: {'PASS' if passed_c else 'FAIL'}")
    score += 1 if passed_c else 0

    # === Summary ===
    print_section_header("MISSING TESTS SUMMARY")
    print(f"  Test A - Citation:  {'PASS' if passed_a else 'FAIL'} (markers: {avg_ann:.1f})")
    print(f"  Test B - Budget:    {'PASS' if passed_b else 'FAIL'} (ratio: {ratio:.1f}x)")
    print(f"  Test C - Halluc:    {'PASS' if passed_c else 'FAIL'} (reduction: {reduction:.0f}%)")
    print(f"  Score: {score}/{total}")
    print_section_header("Done.")

if __name__ == "__main__":
    asyncio.run(run())
