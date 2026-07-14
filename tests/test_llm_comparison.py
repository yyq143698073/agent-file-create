# -*- coding: utf-8 -*-
"""
LLM层对比测试: Critic审查 / 覆盖度提示 / 时效偏好
运行: MODEL=qwen3.5:9b python tests/test_llm_comparison.py

优化(P2/P3/P4):
  P2: 时序测试扩到3个case, 判定收紧(主要结论是否用新值)
  P3: LLM测试3次采样取多数
  P4: 共享config/metrics迁移到test_utils.py
"""
import asyncio, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from agent_file_create.document._critic import run_critic
from tests.test_utils import make_llm, print_section_header, STYLE, MODEL
from tests.test_utils import has_year_citation, prefers_newer, count_placeholder, count_citations

TEST1_MATERIALS = """
[source_a.pdf] RAG系统在NQ数据集上准确率达78.3%，在TriviaQA上达82.1%。
检索阶段使用DPR编码器(dpr-ctx-encoder)，生成阶段使用T5-large。
实验使用4张A100 GPU，推理延迟为120ms/query。
该方法在2023年6月首次提出，代码已开源在GitHub。
[source_b.pdf] 对比实验中，传统BM25检索准确率为62.7%，DPR为78.3%，混合方法为81.5%。
使用8张V100 GPU进行训练，batch size为64。
作者团队来自清华大学和微软亚洲研究院。"""

TEST1_CONTENT_WITH_ERRORS = """
RAG系统在NQ数据集上的准确率达到85.1%，在TriviaQA上达到88.3%。
该方法使用BERT-base作为检索编码器，生成阶段使用GPT-2。
实验使用8张A100 GPU进行训练(source_b提到的是V100)，推理延迟约为200ms。
相比传统BM25检索的52.3%准确率，RAG方法提升了约30个百分点。
该研究由北京大学和Google Research合作完成。"""

KNOWN_ERRORS = [
    "85.1% (correct: 78.3%)", "88.3% (correct: 82.1%)",
    "BERT-base (correct: DPR)", "GPT-2 (correct: T5-large)",
    "200ms (correct: 120ms)", "52.3% BM25 (correct: 62.7%)",
    "北大+Google (correct: 清华+微软)",
]

# Test 2
TEST2_MATERIALS = """
RAG(Retrieval-Augmented Generation)由Lewis等人于2020年提出，结合了密集检索和序列到序列生成。
核心组件包括DPR检索器和BART生成器，在知识密集型任务上显著优于纯生成模型。
实验结果显示，在NQ数据集上EM得分为41.5，在TriviaQA上为56.8。
主要局限包括：检索延迟较高(120ms/query)、对噪声文档敏感、需要大量GPU内存(4张A100)。
未来方向包括更高效的检索索引、多模态RAG、以及减少对标注数据的依赖。"""

COVERAGE_MAP_TEXT = """
以下知识点在当前检索材料中的覆盖情况：
  [充足] RAG核心架构: DPR检索器 + BART生成器
  [充足] 关键实验结果: NQ 41.5 EM, TriviaQA 56.8 EM
  [充足] 主要局限: 检索延迟120ms, 噪声敏感, 4x A100
  [有限] 未来方向: 高效索引, 多模态RAG, 少标注数据
  [无]   与传统方法对比: 纯生成模型的准确率基线
写作时请注意：
- [充足] 的知识点可以深入展开，引用具体数据。
- [有限] 的知识点只写材料中已有的内容，不要延伸推测。
- [无] 的知识点：如果跳过不影响完整性则跳过；
   如果必须提及，用一句话概括并标注[需补充数据]。"""

WRITE_PROMPT_NO = ChatPromptTemplate.from_messages([
    ("human", """撰写一段关于RAG的简要报告，覆盖以下主题：RAG技术原理、关键实验结果、局限性、与传统方法对比。
参考材料: {materials}
要求: 200-400字。"""),
])

WRITE_PROMPT_WITH = ChatPromptTemplate.from_messages([
    ("human", """撰写一段关于RAG的简要报告，覆盖以下主题：RAG技术原理、关键实验结果、局限性、与传统方法对比。
参考材料: {materials}
{coverage_map}
要求: 200-400字。"""),
])

# Test 3: Temporal
TEMPORAL_CASES = [
    {
        "name": "2018 vs 2024(原始)",
        "question": "知识问答任务目前的最佳方法是什么？准确率如何？",
        "materials": """
【1】来源: 2018, old_survey.pdf > 早期研究
2018年的一项综述指出，传统符号推理方法在知识问答任务上的准确率约为45-55%。
【2】来源: 2024, latest_report.pdf > 最新进展
2024年的最新研究表明，基于大语言模型的RAG方法在该任务上的准确率已达82%。
【3】来源: 2019, old_method.pdf > 方法对比
2019年的实验显示，纯检索方法(无生成)的top-1准确率为38%。""",
        "correct_val": "82",
        "newer_year": "2024",
        "older_year": "2018",
    },
    {
        "name": "2020 vs 2024(同指标)",
        "question": "HotpotQA上的最新F1分数是多少？",
        "materials": """
【1】来源: 2020, hotspot_qa.pdf > 实验结果
2020年的实验显示，HotpotQA上的F1分数为67.3。
【2】来源: 2024, rag_progress.pdf > 最新结果
2024年在同一数据集上，改进后的RAG方法取得了F1分数81.7。
【3】来源: 2021, baseline.pdf > 基线对比
2021年的基线方法在HotpotQA上F1为72.1。""",
        "correct_val": "81.7",
        "newer_year": "2024",
        "older_year": "2020",
    },
    {
        "name": "2022 vs 2023 vs 2024(三源)",
        "question": "基于Agent的RAG系统的当前成功率是多少？",
        "materials": """
【1】来源: 2022, agent_survey.pdf > 调查
2022年的调查显示，基于Agent的RAG系统在复杂推理任务上的成功率为63%。
【2】来源: 2023, agent_bench.pdf > 基准测试
2023年的基准测试报告成功率为71%，使用GPT-3.5作为基座模型。
【3】来源: 2024, agent_advance.pdf > 最新突破
2024年最新研究将成功率提升至84%，采用多Agent协作策略。""",
        "correct_val": "84",
        "newer_year": "2024",
        "older_year": "2022",
    },
]

TEMP_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """根据以下材料回答相关问题。
{materials}
请直接回答，2-3句话。"""),
])

def main_answer_uses_newer(text, newer_val):
    cutoff = int(len(text) * 0.4)
    first_part = text[:cutoff]
    markers = ["达到", "达", "取得", "提升至", "实现", "为"]
    for m in markers:
        idx = text.find(m)
        if idx >= 0 and newer_val in text[idx:idx+60]:
            return True
    return newer_val in first_part

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
    return {"samples": samples, "n": n, "n_ok": len(samples), "timings": timings,
            "avg_time": sum(timings)/len(timings) if timings else 0, "label": label}

async def run():
    llm = make_llm(temperature=0.01, max_tokens=800, timeout_s=180)
    cfg = os.getenv("ENDPOINT", "http://localhost:11434")
    print(f"Model: {MODEL} @ {cfg}\n")
    score, total = 0, 0

    # === TEST 1: Critic ===
    print_section_header("TEST 1: Critic LLM Review Quality")
    total += 1
    print("  Known errors (7 total):")
    for e in KNOWN_ERRORS:
        print(f"    - {e}")
    t0 = time.perf_counter()
    report = run_critic(content=TEST1_CONTENT_WITH_ERRORS,
                         outline="## 实验结果\n## 方法细节",
                         materials=TEST1_MATERIALS)
    elapsed = time.perf_counter() - t0
    issues = report.get("issues", [])
    detected = len(issues)
    passed_1 = detected >= 3
    print(f"  Time: {elapsed:.1f}s, Issues: {detected}/7")
    for iss in issues[:5]:
        print(f"    [{iss.get('severity','?')}] {iss.get('description','')}")
    print(f"  Result: {'PASS' if passed_1 else 'FAIL'}")
    score += 1 if passed_1 else 0

    # === TEST 2: Coverage Map (3x sampling) ===
    print_section_header("TEST 2: Coverage Map Effect (3x sampling)")
    total += 1
    r_a = await run_multi(WRITE_PROMPT_NO | llm | StrOutputParser(),
                          {"materials": TEST2_MATERIALS}, n=3, label="no_cov")
    ph_a = [count_placeholder(t) for t in r_a["samples"]]
    ph_avg_a = sum(ph_a)/max(len(ph_a),1)
    r_b = await run_multi(WRITE_PROMPT_WITH | llm | StrOutputParser(),
                          {"materials": TEST2_MATERIALS, "coverage_map": COVERAGE_MAP_TEXT},
                          n=3, label="with_cov")
    ph_b = [count_placeholder(t) for t in r_b["samples"]]
    ph_avg_b = sum(ph_b)/max(len(ph_b),1)
    passed_2 = ph_avg_b >= ph_avg_a
    print(f"  Placeholder avg: {ph_avg_a:.1f} -> {ph_avg_b:.1f}")
    print(f"  Result: {'PASS' if passed_2 else 'FAIL'}")
    score += 1 if passed_2 else 0

    # === TEST 3: Temporal (3 cases x 3 samples) ===
    print_section_header("TEST 3: Temporal Preference (3 cases x 3 samples)")
    total += 1
    chain3 = TEMP_PROMPT | llm | StrOutputParser()
    case_results = []
    for case in TEMPORAL_CASES:
        print(f"\n  [{case['name']}]")
        r = await run_multi(chain3, {"materials": case["materials"]},
                            n=3, label=case["name"])
        texts = r["samples"]
        cv = case["correct_val"]
        ny = case["newer_year"]
        oy = case["older_year"]
        mention_val = sum(1 for t in texts if cv in t)
        mention_ny = sum(1 for t in texts if ny in t)
        newer_first = sum(1 for t in texts if prefers_newer(t, ny, oy))
        answer_newer = sum(1 for t in texts if main_answer_uses_newer(t, cv))
        passed_case = mention_val >= 2 or answer_newer >= 2
        print(f"    val_match={mention_val}/3 newer_first={newer_first}/3 "
              f"answer_uses_newer={answer_newer}/3")
        print(f"    -> {'PASS' if passed_case else 'FAIL'}")
        case_results.append(passed_case)

    passed_3 = sum(case_results) >= 2
    print(f"\n  Overall: {sum(case_results)}/{len(TEMPORAL_CASES)} cases passed")
    score += 1 if passed_3 else 0

    # === Summary ===
    print_section_header("LLM COMPARISON SUMMARY")
    print(f"  Test1-Critic:      {'PASS' if passed_1 else 'FAIL'} ({detected}/7)")
    print(f"  Test2-Coverage:    {'PASS' if passed_2 else 'FAIL'} "
          f"(ph: {ph_avg_a:.1f}->{ph_avg_b:.1f})")
    print(f"  Test3-Temporal:    {'PASS' if passed_3 else 'FAIL'} "
          f"({sum(case_results)}/{len(TEMPORAL_CASES)} cases)")
    print(f"  Score: {score}/{total}")
    print_section_header("Done.")

if __name__ == "__main__":
    asyncio.run(run())
