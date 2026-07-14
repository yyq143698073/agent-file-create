"""Lightweight end-to-end RAG optimization impact evaluation.

Tests how RAG retrieval method + faithfulness fix affects document quality.
Uses paragraph-level answers (200-400 words), minimal LLM calls.

Usage:
  python scripts/eval_rag_impact.py           # 5 built-in questions, ~50 LLM calls
  python scripts/eval_rag_impact.py --all     # 10 questions + downloaded HotpotQA, ~100 calls
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


# ── LLM helper ────────────────────────────────────────────────────────────────

def _call(prompt, timeout_s=15, num_predict=200, temperature=0.0,
           system="你是一个中文文档处理助手。"):
    from agent_file_create.llm_client import call_llm
    r = call_llm(prompt, timeout_s=timeout_s, temperature=temperature,
                 num_predict=num_predict, system=system)
    return r.strip() if isinstance(r, str) and not r.startswith("{") else ""


# ── Built-in test data (multi-hop, paragraph-level) ──────────────────────────

BUILTIN = [
    {"id": "Q1", "type": "bridge",
     "question": "Shirley Temple在电影Kiss and Tell中扮演Corliss Archer，她后来担任了什么政府职务？任职时间是什么？",
     "doc_titles": ["Shirley Temple", "Kiss and Tell", "US Chief of Protocol"],
     "doc_texts": [
         "Shirley Temple (1928-2014) was an American actress, singer, and diplomat. "
         "She served as United States Chief of Protocol from 1976 to 1977. "
         "Temple was appointed Ambassador to Ghana in 1974. She received Kennedy Center Honors in 1998.",
         "Kiss and Tell is a 1945 American comedy film. Shirley Temple starred as Corliss Archer. "
         "The film was based on a popular radio show about teenage life.",
         "The Chief of Protocol is an officer of the US Department of State, responsible for "
         "diplomatic protocol and ceremonies. The position manages visits by foreign dignitaries."],
     "gold_facts": ["Shirley Temple served as US Chief of Protocol",
                    "任职时间是1976年到1977年",
                    "她曾是演员和外交官",
                    "她在Kiss and Tell中扮演Corliss Archer"]},
    {"id": "Q2", "type": "comparison",
     "question": "Arthur's Magazine和First for Women哪个杂志创刊更早？分别在哪年创刊？",
     "doc_titles": ["Arthur's Magazine", "First for Women"],
     "doc_texts": [
         "Arthur's Magazine was an American literary periodical published in the 19th century. "
         "It was first published in 1844 in Philadelphia by Timothy Shay Arthur. "
         "Edgar Allan Poe contributed writings to the magazine.",
         "First for Women is a women's magazine published in the United States. "
         "It was first published in 1989 by Bauer Media Group. "
         "The magazine covers health, diet, and beauty topics with circulation of 4 million."],
     "gold_facts": ["Arthur's Magazine创刊于1844年", "First for Women创刊于1989年",
                    "Arthur's Magazine更早", "两者相差145年"]},
    {"id": "Q3", "type": "bridge",
     "question": "东京大学位于哪个城市？该城市的人口大约是多少？东京大学是哪年成立的？",
     "doc_titles": ["University of Tokyo", "Tokyo"],
     "doc_texts": [
         "The University of Tokyo is a public research university in Bunkyo, Tokyo, Japan. "
         "It was established in 1877 as Japan's first imperial university. "
         "The university has produced numerous Nobel laureates and prime ministers.",
         "Tokyo is the capital city of Japan. As of 2023, Tokyo has a population of "
         "approximately 14 million within its 23 special wards. The Greater Tokyo Area "
         "has over 37 million residents, making it the largest metropolitan area worldwide."],
     "gold_facts": ["东京大学位于东京", "东京人口约1400万(2023年)",
                    "东京大学成立于1877年", "是日本第一所帝国大学"]},
    {"id": "Q4", "type": "bridge",
     "question": "2010年FIFA世界杯决赛的制胜球是谁打进的？他在巴塞罗那职业生涯中进了多少球？",
     "doc_titles": ["2010 World Cup Final", "Andres Iniesta"],
     "doc_texts": [
         "The 2010 FIFA World Cup Final was played between Spain and the Netherlands. "
         "Andres Iniesta scored the winning goal in the 116th minute. "
         "Spain won 1-0, claiming their first World Cup title.",
         "Andres Iniesta is a Spanish former professional footballer. "
         "He spent 16 years at FC Barcelona, scoring 35 goals in 442 appearances. "
         "Iniesta won 9 La Liga titles and 4 Champions League titles with Barcelona."],
     "gold_facts": ["Andres Iniesta打入制胜球", "第116分钟进球",
                    "他在巴萨进了35个球", "西班牙1-0获胜"]},
]

# ── Metrics ───────────────────────────────────────────────────────────────────

def decompose_to_facts(text: str) -> list[str]:
    """Break text into atomic verifiable facts. (1 LLM call)"""
    if len(text) < 30:
        return []
    raw = _call(
        f"将以下文本分解为原子事实(每行一条,每条一句话,不超过30字):\n\n{text[:2000]}\n\n输出:",
        timeout_s=15, num_predict=300)
    return [f.strip() for f in (raw or "").splitlines() if f.strip() and len(f.strip()) > 4][:10]


def verify_facts(facts: list[str], sources: str) -> dict:
    """Verify each fact against source materials using semantic matching. (1 LLM call)"""
    if not facts or not sources:
        return {f: False for f in facts}
    lines = "\n".join(f"[{i+1}] {f}" for i, f in enumerate(facts))
    raw = _call(
        f"请逐条判断以下事实陈述的「含义」是否能在来源材料中找到支撑。\n"
        f"注意：不要逐字匹配，判断语义是否一致即可。\n"
        f"例如「担任礼宾司司长」和「served as Chief of Protocol」语义一致应判OK。\n\n"
        f"来源材料:\n{sources[:4000]}\n\n"
        f"事实陈述:\n{lines}\n\n"
        f"对每条，语义上有支撑回复OK，否则回复MISS。每行一条: 序号 OK/MISS",
        timeout_s=20, num_predict=200)
    result = {}
    for line in (raw or "").splitlines():
        m = re.match(r"(\d+)\s*(OK|MISS)", line.strip(), re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(facts):
                result[facts[idx]] = m.group(2).upper() == "OK"
    return result


def compute_citation_accuracy(text: str, source_docs: list[str]) -> dict:
    """Check if citations in text can be traced to actual sources. (0 LLM calls)"""
    citations = re.findall(r"[（(]据(.+?)[）)]", text)
    if not citations:
        return {"total": 0, "valid": 0, "accuracy": 1.0}
    all_text = " ".join(source_docs).lower()
    valid = 0
    for cite in citations:
        cite_lower = cite.lower().strip()
        # Check if any keyword from the citation appears in source docs
        keywords = [w for w in re.findall(r'[一-鿿\w]{2,}', cite_lower) if w]
        if keywords and sum(1 for kw in keywords if kw in all_text) >= max(1, len(keywords) // 2):
            valid += 1
    return {"total": len(citations), "valid": valid,
            "accuracy": valid / len(citations) if citations else 1.0}


def token_overlap(candidate: str, reference: str) -> float:
    """Token-level F1 overlap with ground truth. (0 LLM calls)"""
    def tok(s): return set(re.findall(r'[一-鿿\w]{2,}', s.lower()))
    c, r = tok(candidate), tok(reference)
    if not c or not r: return 0.0
    overlap = c & r
    p = len(overlap) / len(c) if c else 0
    rec = len(overlap) / len(r) if r else 0
    return 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0


# ── RAG generation ────────────────────────────────────────────────────────────

def rag_generate(kb, kb_name: str, question: str, target_words: int = 250,
                 use_hyde: bool = False) -> tuple[str, int]:
    """Retrieve from KB and generate a paragraph. Returns (text, llm_call_count)."""
    calls = 0

    # Step 1: Retrieve
    if use_hyde:
        hits = kb.search_hyde(kb=kb_name, query=question, top_k=4)
    else:
        hits = kb.search(kb=kb_name, query=question, top_k=4)
    if not hits:
        hits = kb.search(kb=kb_name, query=question, top_k=4)  # fallback

    # Step 2: Build context
    ctx = ""
    seen = set()
    for h in hits[:5]:
        if h.chunk_id in seen: continue
        seen.add(h.chunk_id)
        chunk = str(h.content or "")[:400]
        if chunk:
            ctx += f"[来源:{h.doc_id or '未知'}] {chunk}\n\n"

    if not ctx and not use_hyde:
        # Try HyDE as fallback
        return rag_generate(kb, kb_name, question, target_words, use_hyde=True)

    # Step 3: Generate answer
    calls += 1
    gen_prompt = (
        f"你是一个文档处理助手。基于以下检索到的材料回答用户问题。\n\n"
        f"要求：\n- 回答200-300字，简洁准确\n"
        f"- 每个关键事实后标注来源，如（据文档名）\n"
        f"- 如果材料不足以回答，明确说明\n"
        f"- 不要编造材料中没有的信息\n\n"
        f"检索材料:\n{ctx[:2500]}\n\n"
        f"用户问题: {question}\n\n回答:"
    )
    answer = _call(gen_prompt, timeout_s=30, num_predict=400, temperature=0.2)

    if use_hyde:
        calls += 1  # HyDE itself costs 1 LLM call

    return (answer or "").strip(), calls


def run_faithfulness_fix(content: str, sources: str, task_id: str) -> tuple[str, int]:
    """Lightweight faithfulness fix: detect unsupported claims and remove them.
    Returns (fixed_content, llm_call_count)."""
    calls = 0
    if not sources.strip() or len(content) < 50:
        return content, 0

    # Step 1: Quick check for unsupported claims (1 LLM call)
    calls += 1
    check_raw = _call(
        f"你是文档事实核查助手。检查以下段落中是否有无依据的断言。\n\n"
        f"来源材料:\n{sources[:3000]}\n\n段落:\n{content[:1500]}\n\n"
        f"如果所有内容都有依据,回复ALL_OK。如果有问题,逐条列出: WARN: <问题描述>",
        timeout_s=15, num_predict=200)

    if not check_raw or "ALL_OK" in check_raw or "无" in check_raw:
        return content, calls

    # Step 2: Conservative fix — only correct errors, preserve correct content
    calls += 1
    fix_raw = _call(
        f"你是一个谨慎的文档编辑。请修正以下段落中的事实错误，规则如下：\n"
        f"1. 保留原文结构和措辞，只修改错误的、与检索证据冲突的部分\n"
        f"2. 仅在检索证据直接支持时修改数字、名称、日期\n"
        f"3. 不要重写或简化段落，不要删除已有的正确信息\n"
        f"4. 保持同样的细节水平\n\n"
        f"原始段落:\n{content[:1500]}\n\n"
        f"核查发现的问题:\n{check_raw[:500]}\n\n"
        f"修正后的段落（只改有问题的部分，保留其余所有内容）:",
        timeout_s=15, num_predict=400, temperature=0.1)
    return (fix_raw or content).strip(), calls


# ── Main ──────────────────────────────────────────────────────────────────────

def run(num_questions: int = 5, use_hotpotqa: bool = False):
    from agent_file_create.rag.kb import KnowledgeBase

    # Load data
    if use_hotpotqa:
        try:
            from scripts.eval_hotpotqa import download_hotpotqa
            queries = download_hotpotqa(num_questions)
            print(f"Loaded {len(queries)} HotpotQA questions")
        except Exception as e:
            print(f"Download failed ({e}), using built-in")
            queries = BUILTIN[:num_questions]
    else:
        queries = BUILTIN[:num_questions]

    # Build KB
    kb = KnowledgeBase()
    kb_name = "eval_rag_impact"
    try:
        kb.delete_kb(kb=kb_name)
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        doc_map = OrderedDict()
        for q in queries:
            for title, text in zip(q.get("doc_titles", []), q.get("doc_texts", [])):
                if title not in doc_map:
                    doc_map[title] = text
        for title, text in doc_map.items():
            safe = re.sub(r'[^a-zA-Z0-9一-鿿]', '_', title)[:60] + '.txt'
            (tmp / safe).write_text(text, encoding='utf-8')
        for fp in sorted(tmp.iterdir()):
            kb.ingest_file(kb=kb_name, file_path=str(fp))

    all_sources = "\n\n".join(doc_map.values())
    print(f"KB ready: {len(doc_map)} docs, {len(queries)} questions\n")
    print(f"{'#'  :<4} {'Q':<5} {'方法':<12} {'忠实度':>8} {'覆盖率':>8} {'引用':>8} {'LLM':>5} {'秒':>5}")
    print("-" * 65)

    results = {"baseline": [], "optimized": []}
    total_calls = 0

    for i, q in enumerate(queries):
        question = q["question"]
        gold_facts_text = "；".join(q.get("gold_facts", []))
        sources_for_q = "\n\n".join(q.get("doc_texts", []))

        for method, label in [("baseline", "search"), ("optimized", "hyde+fix")]:
            use_hyde = (method == "optimized")
            t0 = time.perf_counter()

            # Generate
            answer, gen_calls = rag_generate(kb, kb_name, question, use_hyde=use_hyde)
            fix_calls = 0

            # Faithfulness fix (optimized only)
            if method == "optimized":
                answer, fix_calls = run_faithfulness_fix(answer, sources_for_q, f"eval_{i}")

            elapsed = time.perf_counter() - t0
            llm_used = gen_calls + fix_calls
            total_calls += llm_used

            # Evaluate
            facts = decompose_to_facts(answer)
            verification = verify_facts(facts, sources_for_q) if facts else {}
            faithful = sum(1 for v in verification.values() if v)
            faithful_rate = faithful / len(facts) if facts else 0.0

            # Coverage: LLM judges whether each gold fact is covered (1 LLM call, batched)
            gold_facts = [g.strip() for g in gold_facts_text.split("；") if g.strip()]
            if gold_facts and answer:
                gold_lines = "\n".join(f"[{i+1}] {g}" for i, g in enumerate(gold_facts))
                cov_raw = _call(
                    f"逐条判断以下关键信息点是否在生成的段落中有所体现。\n"
                    f"注意：同义表达、改写、不同措辞都算覆盖，不要求逐字一致。\n\n"
                    f"关键信息点:\n{gold_lines}\n\n生成段落:\n{answer[:2000]}\n\n"
                    f"每行一条: 序号 YES/NO（YES=段落中已包含该信息，NO=完全没有提及）",
                    timeout_s=12, num_predict=100)
                gold_covered = 0
                for line in (cov_raw or "").splitlines():
                    m = re.match(r"(\d+)\s*(YES|NO)", line.strip(), re.IGNORECASE)
                    if m and m.group(2).upper() == "YES":
                        gold_covered += 1
                gold_total = len(gold_facts)
                coverage = gold_covered / gold_total
            else:
                coverage = 0.0

            # Citation accuracy
            cit = compute_citation_accuracy(answer, q.get("doc_texts", []))

            results[method].append({
                "faithfulness": round(faithful_rate, 3),
                "coverage": round(coverage, 3),
                "citation_accuracy": round(cit["accuracy"], 3),
                "llm_calls": llm_used,
                "elapsed_s": round(elapsed, 1),
            })

            print(f"  {i+1:<4} {q['id']:<5} {label:<12} {faithful_rate:>8.3f} "
                  f"{coverage:>8.3f} {cit['accuracy']:>8.3f} {llm_used:>5} {elapsed:>5.1f}")

    # ── Aggregate ──
    def avg(lst, key):
        return round(sum(r[key] for r in lst) / max(len(lst), 1), 3)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'指标':<25} {'search(基线)':>12} {'hyde+fix(优化)':>14} {'改善':>10}")
    print("-" * 65)

    for metric, label in [
        ("faithfulness", "忠实度"),
        ("coverage", "覆盖率"),
        ("citation_accuracy", "引用准确率"),
    ]:
        b = avg(results["baseline"], metric)
        o = avg(results["optimized"], metric)
        imp = o - b
        print(f"  {label:<25} {b:>12.3f} {o:>14.3f} {imp:>+10.3f}")

    b_calls = sum(r["llm_calls"] for r in results["baseline"])
    o_calls = sum(r["llm_calls"] for r in results["optimized"])
    b_time = sum(r["elapsed_s"] for r in results["baseline"])
    o_time = sum(r["elapsed_s"] for r in results["optimized"])

    print(f"  {'LLM调用总数':<25} {b_calls:>12} {o_calls:>14} {o_calls-b_calls:>+10}")
    print(f"  {'总耗时(秒)':<25} {b_time:>12.1f} {o_time:>14.1f} {o_time-b_time:>+10.1f}")
    print()
    print(f"  测试题目数: {len(queries)}")
    print(f"  总LLM调用: {total_calls} 次")
    print(f"  总耗时: {b_time+o_time:.0f} 秒")

    # Save
    out = _PROJ / "result" / "eval_rag_impact.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"num_questions": len(queries), "method": "A/B test"},
        "aggregate": {
            "baseline": {k: avg(results["baseline"], k) for k in
                         ["faithfulness", "coverage", "citation_accuracy"]},
            "optimized": {k: avg(results["optimized"], k) for k in
                          ["faithfulness", "coverage", "citation_accuracy"]},
            "baseline_llm_calls": b_calls,
            "optimized_llm_calls": o_calls,
            "total_llm_calls": total_calls,
        },
        "per_question": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结果: {out}")

    try:
        kb.delete_kb(kb=kb_name)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG optimization impact eval")
    parser.add_argument("--num", type=int, default=4, help="Number of questions (default: 4)")
    parser.add_argument("--all", action="store_true", help="Use all built-in + try HotpotQA")
    args = parser.parse_args()
    run(num_questions=args.num, use_hotpotqa=args.all)
