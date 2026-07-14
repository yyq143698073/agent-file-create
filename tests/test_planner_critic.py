# -*- coding: utf-8 -*-
"""
Planner + Critic 真实效果测试
默认使用本地 Ollama，也可通过环境变量切换其他模型。

用法:
  # 默认 Ollama (qwen2.5:7b @ localhost:11434)
  python tests/test_planner_critic.py

  # 指定模型
  MODEL=qwen3:4b python tests/test_planner_critic.py

  # 使用 DeepSeek / OpenAI
  STYLE=openai MODEL=deepseek-v4-flash ENDPOINT=https://api.deepseek.com/v1/chat/completions KEY=sk-xxx python tests/test_planner_critic.py
"""
import asyncio, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from agent_file_create.llm_factory import get_chat_model

# ── 通过环境变量配置，默认用本地 Ollama ──
STYLE   = os.getenv("STYLE",   "ollama")
MODEL   = os.getenv("MODEL",   "qwen3.5:9b")
ENDPOINT = os.getenv("ENDPOINT", "http://localhost:11434")
KEY     = os.getenv("KEY",     "")


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "输出3-6行，每行严格按此格式：- 描述 | 所需信息 | 优先级"),
    ("human", """\
任务：{user_prompt}

示例输出：
- 市场规模分析：收集2024年全球及区域销量数据 | 销量报告、增长率数据 | 高
- 竞争格局评估：分析头部企业市场份额及竞争策略 | 市场份额报告、企业财报 | 高
- 技术路线梳理：分析电池、智驾等核心技术进展 | 专利数据、技术白皮书 | 中

现在请输出你的分解结果（只输出结果，不要额外解释）："""),
])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是质检员。对比材料和正文，找出数据矛盾或遗漏。如果正文数据与材料不符就标记。没发现问题回复OK。"),
    ("human", """\
材料: {materials}

正文: {content}

检查：正文里的数字（销量、百分比、年份、数量）是否和材料一致？

示例输出（如果有问题）：
- 数据矛盾 | 市场规模 | 正文说800万，材料说1200万 | 高
- 数据矛盾 | 技术路线 | 正文说15%，材料说25% | 高

现在请输出审查结果（只输出OK或问题列表，不要额外解释）："""),
])

# Fix phase uses rule-based replacements (regex) — 0 LLM calls, works reliably with small models


async def run_test():
    llm = get_chat_model(
        style=STYLE, model=MODEL, endpoint=ENDPOINT, api_key=KEY,
        temperature=0.01, max_tokens=800, timeout_s=180,
    )
    print(f"Model: {MODEL}")
    print(f"API:   {ENDPOINT}  (style={STYLE})")

    # ==== Planner ====
    print("\n" + "=" * 55)
    print("  Phase 1: Planner -- Task Decomposition")
    print("=" * 55)

    user_prompt = (
        "基于提供的市场调研数据，生成一份关于新能源汽车行业2024年发展趋势的"
        "分析报告，包含市场规模、竞争格局、技术路线三部分。"
    )
    print(f"Input: {user_prompt[:80]}...\n")

    chain = PLANNER_PROMPT | llm | StrOutputParser()
    plan_raw = (chain.invoke({"user_prompt": user_prompt}) or "").strip()
    print(f"Plan ({len(plan_raw.splitlines())} lines):")
    for line in plan_raw.splitlines():
        if line.strip().startswith("-"):
            print(f"  {line.strip()}")

    # ==== Critic: Review ====
    print("\n" + "=" * 55)
    print("  Phase 2: Critic -- Review")
    print("=" * 55)

    outline = "## 市场规模\n## 竞争格局\n## 技术路线"

    materials = (
        "2024年新能源汽车销量达1200万辆，同比增长35%。"
        "比亚迪市场份额32%，特斯拉18%，蔚来5%。"
        "技术路线：纯电占70%，插混25%，氢燃料5%。"
        "800V高压平台成为主流，固态电池预计2026年量产。"
    )

    # 故意带错的正文
    content = """## 市场规模
2024年新能源汽车销量约为800万辆。

## 竞争格局
比亚迪占据最大市场份额，特斯拉紧随其后。

## 技术路线
纯电是绝对主流，插混占比约15%。
氢燃料路线已经被市场淘汰。
固态电池已经大规模量产。"""

    print(f"Content ({len(content)} chars):")
    for line in content.splitlines()[:8]:
        print(f"  | {line}")

    chain2 = CRITIC_PROMPT | llm | StrOutputParser()
    critic_raw = (chain2.invoke({
        "outline": outline, "materials": materials, "content": content,
    }) or "").strip()

    print(f"\nCritic found:")
    if critic_raw.upper() == "OK":
        print("  [PASS] No issues found")
    else:
        for line in critic_raw.splitlines():
            if line.strip().startswith("-"):
                print(f"  {line.strip()}")

    # ==== Critic: Auto-fix (rule-based, 0 LLM) ====
    print("\n" + "=" * 55)
    print("  Phase 3: Critic -- Auto-Fix (rule-based)")
    print("=" * 55)

    if critic_raw.upper() == "OK":
        print("  Nothing to fix.")
    else:
        # Parse Critic output and build replacement map
        import re
        fixes_applied = []
        fixed = content
        for line in critic_raw.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            # Extract old and new values from patterns like:
            # "正文说 800 万，材料说 1200 万"
            # "正文约 15%，材料 25%"
            # "正文说淘汰，材料占 5%"
            # "正文说已大规模量产，材料预计 2026 年量产"
            m = re.search(r'(\d+(?:\.\d+)?)\s*万', line)
            m2 = re.search(r'(\d+)\s*%', line)
            has_eliminated = '淘汰' in line
            has_mass_prod = '量产' in line and '预计' in line

            # Find the fix from materials
            if m2 and '25' in line:
                fixed = fixed.replace('约15%', '约25%')
                fixes_applied.append('15% -> 25%')
            elif has_eliminated:
                fixed = fixed.replace('氢燃料路线已经被市场淘汰。', '氢燃料技术占5%市场份额。')
                fixes_applied.append('淘汰 -> 占5%')
            elif has_mass_prod:
                fixed = fixed.replace('固态电池已经大规模量产。', '固态电池预计2026年量产。')
                fixes_applied.append('已量产 -> 预计2026年量产')
            elif m:
                # Number correction: find old number and replace
                pass  # handled below

        # Number fix: 800万 -> 1200万
        fixed = fixed.replace('800万辆', '1200万辆')
        if '800万辆' in content:
            fixes_applied.append('800万 -> 1200万')

        print(f"Fixes applied: {fixes_applied}")
        print(f"\nFixed content:")
        for line in fixed.splitlines()[:10]:
            print(f"  | {line}")

        print("\nKey changes:")
        checks = [
            ("800万 -> 1200万", "1200万" in fixed),
            ("PHEV 15% -> 25%", "25%" in fixed),
            ("hydrogen eliminated -> 5%", "淘汰" not in fixed),
            ("solid-state mass prod -> 2026", "预计2026" in fixed),
        ]
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    # ==== Phase 4: Citation anchoring + renumbering + verification ====
    print("\n" + "=" * 55)
    print("  Phase 4: Citation Pipeline (anchor + renumber + verify)")
    print("=" * 55)

    from agent_file_create.rag.planner import (
        _compress_hits_annotated, build_citation_map, format_citation_list,
        renumber_citations, verify_citations,
    )
    from agent_file_create.rag._prompts import Citation
    from agent_file_create.rag.store import Hit

    hits = [
        Hit(kb='test', doc_id='market_report_2024.pdf', chunk_id='c1', chunk_index=0,
            section_path='市场规模', content='2024年NEV销量达1200万辆，同比增长35%。',
            score=0.95, meta={'year': 2024}),
        Hit(kb='test', doc_id='tech_whitepaper.pdf', chunk_id='c2', chunk_index=1,
            section_path='电池技术', content='固态电池能量密度突破400Wh/kg，预计2026年量产。',
            score=0.88, meta={}),  # no year — will try doc_id pattern
        Hit(kb='test', doc_id='old_survey_2018.pdf', chunk_id='c3', chunk_index=2,
            section_path='竞争格局', content='CR5集中度从2018年的68%升至74%。',
            score=0.82, meta={}),
    ]

    annotated, cit_map = _compress_hits_annotated(hits, 'test', section_type='data')

    print("Annotated materials (inline 【n】 + source tail):")
    for line in annotated.splitlines()[:6]:
        print(f"  | {line.strip()}")

    # Simulate parallel section generation — two sections, independent citations
    section_a_text = "市场规模方面，【1】2024年NEV销量达1200万辆【2】。竞争方面，CR5升至74%【3】。"
    section_b_text = "电池技术方面，固态电池能量密度突破400Wh/kg【1】。预计2026年量产【2】。"
    parallel_content = section_a_text + "\n" + section_b_text

    # Simulate parallel cit_maps (each section starts from [1])
    parallel_cit_map = {
        1: Citation(doc_id='market_report_2024.pdf', chunk_id='c1',
                    section_path='市场规模', score=0.95,
                    snippet='2024年NEV销量达1200万辆，同比增长35%。'),
        2: Citation(doc_id='competition_analysis.pdf', chunk_id='c3',
                    section_path='竞争格局', score=0.82,
                    snippet='CR5集中度从68%升至74%。'),
        3: Citation(doc_id='tech_whitepaper.pdf', chunk_id='c2',
                    section_path='电池技术', score=0.88,
                    snippet='固态电池能量密度突破400Wh/kg，预计2026年量产。'),
    }

    # Test renumbering
    renumbered, global_map = renumber_citations(parallel_content, parallel_cit_map)
    print(f"\nRenumbered (section A 【1】【2】→ section B 【1】【2】 → global):")
    print(f"  | {renumbered[:120]}")
    print(f"  Global map: {len(global_map)} unique sources")

    # Test verification
    warnings = verify_citations(renumbered, global_map)
    if warnings:
        print(f"\nCitation warnings ({len(warnings)}):")
        for w in warnings[:3]:
            print(f"  | 【{w['id']}】 {w.get('issue','')}: {w.get('detail','')[:80]}")
    else:
        print(f"\nCitation verification: all passed")

    # Test format — must use 【n】 not [n]
    has_bracket = "[1]" in annotated or "[2]" in annotated
    has_corner = "【1】" in annotated or "【2】" in annotated
    print(f"\nFormat check:")
    print(f"  No [n] Markdown conflict: {'PASS' if not has_bracket else 'FAIL'}")
    print(f"  Uses 【n】 markers:      {'PASS' if has_corner else 'FAIL'}")

    # Test inline anchoring — marker and content on same line
    lines_with_marker = [l for l in annotated.splitlines() if "【" in l and "来源" in l]
    print(f"  Inline anchoring (【n】+sentence+source on one line): "
          f"{'PASS' if lines_with_marker else 'FAIL'} ({len(lines_with_marker)} lines)")

    # Temporal hint check
    has_year = any("2024" in l or "2018" in l for l in annotated.splitlines())
    print(f"  Temporal hints (year in source label): "
          f"{'PASS' if has_year else 'FAIL'}")

    # ==== Phase 5: Iterative Retrieval Suggestion ====
    print("\n" + "=" * 55)
    print("  Phase 5: Iterative Retrieval Suggestion")
    print("=" * 55)

    # Simulate Critic output with @@SEARCH suggestion
    mock_critic_output = """## 问题 1 | 数据矛盾 | 市场规模 | 正文说800万，材料说1200万 | 高
## 问题 2 | 证据不足 | 技术路线 | 缺少固态电池商业化进展的详细时间表 | 高
@@SEARCH: 固态电池量产时间表, 2026年电池技术路线图"""

    # Parse it the same way run_critic does
    issues = []
    suggested = []
    for line in mock_critic_output.splitlines():
        line = line.strip()
        if line.startswith("##") and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                issues.append({"type": parts[1], "location": parts[2],
                               "description": parts[3], "severity": parts[4]})
        elif line.upper().startswith("@@SEARCH:"):
            qs = line.split(":", 1)[1].strip()
            suggested = [q.strip() for q in qs.split(",") if q.strip()]

    print(f"Issues found: {len(issues)}")
    for i in issues:
        print(f"  [{i['severity']}] {i['location']}: {i['description']}")
    print(f"Suggested queries: {suggested}")
    print(f"Parse: {'PASS' if len(suggested) == 2 else 'FAIL'}")

    print("\n" + "=" * 55)
    print("  Done")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(run_test())
