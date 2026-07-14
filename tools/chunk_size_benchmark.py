#!/usr/bin/env python
"""Chunk‑size sensitivity benchmark — fixed method, variable sizes.

Tests one chunking method at chunk_size ∈ {200, 500, 1000},
measuring retrieval metrics + optional LLM answer quality.

Usage::

    # Retrieval only (no API needed)
    python tools/chunk_size_benchmark.py --dir ./test_doc/pdf

    # With LLM answer quality evaluation
    python tools/chunk_size_benchmark.py --dir ./test_doc/pdf --llm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure `tools/` is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse extraction from chunk_benchmark
from chunk_benchmark import (
    _extract_text,
    _embed_chunks_tfidf,
    _cosine_sim,
    _keyword_relevance,
)

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
CHUNK_SIZES = [200, 500, 1000]
DEFAULT_METHOD = "recursive"
TOP_K_CONTEXT = 5

QUERY_SETS: Dict[str, List[str]] = {
    "pdf": [
        "RAG技术架构的核心组件有哪些？",
        "知识库构建的关键步骤是什么？",
        "大语言模型在RAG中如何发挥作用？",
    ],
    "docx": [
        "AI Agent的核心特征是什么？",
        "当前AI Agent的市场部署态势如何？",
        "GPT-4o在多模态任务上的表现如何？",
    ],
    "xlsx": [
        "运动训练计划的强度如何递增？",
        "算法筛选的重点指标有哪些？",
        "基线信息中统计了哪些维度？",
    ],
    "jpg": [
        "RAG在线流程包含哪些步骤？",
        "文档切分后如何向量化？",
        "Query预处理有哪些方法？",
    ],
    "default": [
        "请总结文档的核心内容",
        "文档中提到的关键技术有哪些？",
        "文档得出的主要结论是什么？",
    ],
}

_SUPPORTED_SUFFIXES = {
    ".txt", ".md", ".markdown", ".csv", ".json",
    ".pdf", ".docx", ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Recursive chunking (copied for standalone use, param tweaks)
# ═══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


_RECURSIVE_SEPARATORS = [
    "\n\n", "\n",
    r"(?<=[。！？])(?=\S)",
    r"(?<=[.!?])\s+(?=\S)",
    r"(?<=[；;])(?=\S)",
    r"(?<=[，,;])(?=\S)",
    " ",
]


def _recursive_split(text: str, separators: List[str], target: int) -> List[str]:
    if len(text) <= target:
        return [text] if text.strip() else []
    for sep in separators:
        if isinstance(sep, str) and sep.startswith(r"(?<="):
            parts = re.split(sep, text)
        elif sep in text:
            parts = text.split(sep)
        else:
            continue
        good = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= target:
                good.append(part)
            else:
                good.extend(_recursive_split(part, separators[1:], target))
        if good:
            return good
    return [text[i:i + target] for i in range(0, len(text), target)]


def chunk_recursive(text: str, chunk_size: int = 500) -> List[str]:
    if not text.strip():
        return []
    return [_norm(c) for c in _recursive_split(text, _RECURSIVE_SEPARATORS, chunk_size)]


# ═══════════════════════════════════════════════════════════════════════════════
# LLM integration (standalone, reads config from env / args)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_llm_client(api_style: str, api_endpoint: str, api_key: str, model: str):
    """Build an OpenAI‑compatible chat client."""
    from openai import OpenAI
    if api_style in ("openai", "ollama", "vllm", ""):
        base_url = api_endpoint or None
        if api_style == "ollama" and base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        return OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    raise ValueError(f"Unsupported API style: {api_style}")


def _generate_answer(
    query: str,
    context_chunks: List[str],
    *,
    api_style: str,
    api_endpoint: str,
    api_key: str,
    model: str,
    timeout_s: int = 120,
) -> Tuple[str, float]:
    """Ask LLM to answer *query* based on *context_chunks*. Returns (answer, latency_s)."""
    from openai import OpenAI

    client = _build_llm_client(api_style, api_endpoint, api_key, model)
    context = "\n\n---\n\n".join(context_chunks[:TOP_K_CONTEXT])

    system = (
        "你是一个文档分析助手。请严格根据提供的上下文回答问题。"
        '如果上下文中没有相关信息，请明确说"未找到相关信息"。'
        "回答要简洁、准确，不超过300字。"
    )
    user = f"上下文：\n{context}\n\n问题：{query}"

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=512,
            timeout=timeout_s,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as exc:
        answer = f"[LLM调用失败: {exc}]"
    latency = time.perf_counter() - t0
    return answer, latency


def _evaluate_answer(
    query: str,
    answer: str,
    context_chunks: List[str],
    *,
    api_style: str,
    api_endpoint: str,
    api_key: str,
    model: str,
    timeout_s: int = 60,
) -> Dict[str, float]:
    """LLM‑as‑judge: score answer on relevance / completeness / factual (1-5 each)."""
    from openai import OpenAI

    client = _build_llm_client(api_style, api_endpoint, api_key, model)
    context = "\n\n---\n\n".join(context_chunks[:TOP_K_CONTEXT])

    judge_prompt = textwrap.dedent(f"""\
    你是一个严格的评测员。请根据以下上下文评估答案质量。

    【上下文】
    {context}

    【问题】
    {query}

    【待评估答案】
    {answer}

    请从以下三个维度打分（1-5分），只输出JSON：
    {{
      "relevance": <1-5, 答案与问题的相关度>,
      "completeness": <1-5, 答案覆盖上下文关键信息的完整度>,
      "factual": <1-5, 答案与上下文事实的一致性, 如有编造则低分>
    }}
    """)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=256,
            timeout=timeout_s,
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception:
        return {"relevance": 0, "completeness": 0, "factual": 0}

    # Parse JSON
    try:
        raw = re.sub(r"```json\s*|```", "", raw).strip()
        scores = json.loads(raw)
        return {
            "relevance": float(scores.get("relevance", 0)),
            "completeness": float(scores.get("completeness", 0)),
            "factual": float(scores.get("factual", 0)),
        }
    except Exception:
        return {"relevance": 0, "completeness": 0, "factual": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SizeResult:
    chunk_size: int
    chunk_count: int = 0
    avg_length: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    retrieval_ms: float = 0.0
    embed_ms: float = 0.0
    # LLM
    llm_relevance: float = 0.0
    llm_completeness: float = 0.0
    llm_factual: float = 0.0
    llm_latency_s: float = 0.0
    answers: List[str] = field(default_factory=list)


def _find_queries(file_paths: List[Path]) -> List[str]:
    """Auto‑detect the best query set from file directory names."""
    dir_names = {p.parent.name.lower() for p in file_paths}
    for key in ["pdf", "docx", "xlsx", "jpg"]:
        if key in dir_names:
            return QUERY_SETS[key]
    return QUERY_SETS["default"]


def run_size_benchmark(
    texts: Dict[str, str],
    queries: List[str],
    *,
    chunk_sizes: List[int] = None,
    llm_config: Optional[Dict[str, str]] = None,
) -> List[SizeResult]:
    """Run the chunk‑size sweep."""
    if chunk_sizes is None:
        chunk_sizes = CHUNK_SIZES

    combined = "\n\n".join(texts.values())
    results: List[SizeResult] = []

    for cs in chunk_sizes:
        print(f"\n{'='*60}")
        print(f"  chunk_size = {cs}")
        print(f"{'='*60}")

        r = SizeResult(chunk_size=cs)

        # ── Chunking ──────────────────────────────────────────────────
        chunks = chunk_recursive(combined, chunk_size=cs)
        r.chunk_count = len(chunks)
        if chunks:
            lengths = [len(c) for c in chunks]
            r.avg_length = sum(lengths) / len(chunks)
        print(f"  块数: {r.chunk_count}, 平均长度: {r.avg_length:.0f} chars")

        # ── Retrieval ─────────────────────────────────────────────────
        if not chunks:
            results.append(r)
            continue

        t0 = time.perf_counter()
        vec, chunk_vecs = _embed_chunks_tfidf(chunks)
        r.embed_ms = (time.perf_counter() - t0) * 1000

        import numpy as np
        t0 = time.perf_counter()
        query_vecs = vec.transform(queries)
        sims = _cosine_sim(query_vecs, chunk_vecs)
        r.retrieval_ms = (time.perf_counter() - t0) * 1000

        # Recall / MRR via keyword pseudo‑relevance
        recalls_3, recalls_5, mrrs = [], [], []
        for qi, (query, row) in enumerate(zip(queries, sims)):
            relevant = _keyword_relevance(query, chunks)
            if not relevant:
                continue
            ranked = np.argsort(-row)
            for k, store in [(3, recalls_3), (5, recalls_5)]:
                hits = len(set(ranked[:k]) & relevant)
                store.append(hits / min(k, len(relevant)))
            for rank, idx in enumerate(ranked, start=1):
                if idx in relevant:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)

        r.recall_at_3 = float(np.mean(recalls_3)) if recalls_3 else 0.0
        r.recall_at_5 = float(np.mean(recalls_5)) if recalls_5 else 0.0
        r.mrr = float(np.mean(mrrs)) if mrrs else 0.0
        print(f"  R@3={r.recall_at_3:.3f}  R@5={r.recall_at_5:.3f}  MRR={r.mrr:.3f}")

        # ── LLM answer quality ────────────────────────────────────────
        if llm_config:
            print("  LLM 回答生成 + 评估中...")
            rel_scores, comp_scores, fact_scores = [], [], []
            total_latency = 0.0
            for qi, query in enumerate(queries):
                # Retrieve top‑K chunks
                row = sims[qi]
                ranked = np.argsort(-row)
                top_indices = ranked[:TOP_K_CONTEXT]
                context = [chunks[i] for i in top_indices]

                answer, lat = _generate_answer(query, context, **llm_config)
                total_latency += lat
                r.answers.append(answer)

                scores = _evaluate_answer(query, answer, context, **llm_config)
                rel_scores.append(scores["relevance"])
                comp_scores.append(scores["completeness"])
                fact_scores.append(scores["factual"])

                print(f"    Q{qi+1}: rel={scores['relevance']:.0f} "
                      f"comp={scores['completeness']:.0f} "
                      f"fact={scores['factual']:.0f} "
                      f"({lat:.1f}s)")

            r.llm_relevance = float(np.mean(rel_scores)) if rel_scores else 0.0
            r.llm_completeness = float(np.mean(comp_scores)) if comp_scores else 0.0
            r.llm_factual = float(np.mean(fact_scores)) if fact_scores else 0.0
            r.llm_latency_s = total_latency

        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(results: List[SizeResult], *, has_llm: bool = False) -> None:
    """Print Markdown comparison table."""
    headers = ["chunk_size", "块数", "平均长度", "R@3", "R@5", "MRR", "嵌入ms", "检索ms"]
    if has_llm:
        headers += ["相关性", "完整度", "事实性", "LLM耗时s"]

    widths = [12, 8, 10, 8, 8, 8, 10, 10]
    if has_llm:
        widths += [8, 8, 8, 10]

    def _hdr() -> str:
        h = "| " + " | ".join(f"{c:^{w}}" for c, w in zip(headers, widths)) + " |"
        s = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        return h + "\n" + s

    def _row(*cols) -> str:
        return "| " + " | ".join(
            f"{c:^{w}}" for c, w in zip(cols, widths)
        ) + " |"

    print()
    print(_hdr())
    for r in results:
        cells = [
            str(r.chunk_size),
            str(r.chunk_count),
            f"{r.avg_length:.0f}",
            f"{r.recall_at_3:.3f}",
            f"{r.recall_at_5:.3f}",
            f"{r.mrr:.3f}",
            f"{r.embed_ms:.0f}",
            f"{r.retrieval_ms:.0f}",
        ]
        if has_llm:
            cells += [
                f"{r.llm_relevance:.1f}",
                f"{r.llm_completeness:.1f}",
                f"{r.llm_factual:.1f}",
                f"{r.llm_latency_s:.1f}",
            ]
        print(_row(*cells))

    print()
    print("指标说明：")
    print("  R@K / MRR = 检索命中率（bigram 关键词伪相关）")
    if has_llm:
        print("  相关性/完整度/事实性 = LLM 自评分数（1-5），越高越好")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Chunk‑size sensitivity benchmark — variable size, fixed method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          python tools/chunk_size_benchmark.py --dir ./test_doc/pdf
          python tools/chunk_size_benchmark.py --dir ./test_doc/pdf --llm
          python tools/chunk_size_benchmark.py --dir ./test_doc/docx --sizes 200 500 1000
          python tools/chunk_size_benchmark.py --files a.pdf b.pdf --query "Q1" "Q2" "Q3"
        """),
    )
    ap.add_argument("--dir", help="Directory containing documents")
    ap.add_argument("--files", nargs="+", default=[], help="Specific files")
    ap.add_argument("--query", nargs="+", default=[], help="Test queries (3 recommended)")
    ap.add_argument("--sizes", nargs="+", type=int, default=CHUNK_SIZES,
                    help=f"Chunk sizes to test (default: {CHUNK_SIZES})")
    ap.add_argument("--llm", action="store_true",
                    help="Enable LLM answer generation + quality evaluation")
    ap.add_argument("--api-style", default="ollama",
                    help="API style: openai / ollama (default: ollama)")
    ap.add_argument("--api-endpoint", default="",
                    help="API endpoint URL")
    ap.add_argument("--api-key", default="",
                    help="API key")
    ap.add_argument("--model", default="",
                    help="Model name")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # ── Gather files ──────────────────────────────────────────────────
    file_paths: List[Path] = [Path(f) for f in args.files]
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"错误: 目录不存在 — {args.dir}")
            sys.exit(1)
        file_paths.extend(sorted(dir_path.rglob("*")))
    file_paths = [p for p in file_paths if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES]
    file_paths = list(dict.fromkeys(file_paths))
    if not file_paths:
        print("错误: 未找到可处理的文件。支持的格式:", ", ".join(sorted(_SUPPORTED_SUFFIXES)))
        sys.exit(1)

    # ── Extract ───────────────────────────────────────────────────────
    texts: Dict[str, str] = {}
    total_chars = 0
    print(f"正在从 {len(file_paths)} 个文件提取文本...")
    for fp in file_paths:
        content = _extract_text(str(fp))
        if content:
            texts[fp.name] = content
            total_chars += len(content)
            print(f"  ✓ {fp.name} ({len(content)} chars)")
        else:
            print(f"  ✗ {fp.name} (无内容)")
    if not texts:
        print("错误: 所有文件均无可提取的文本。")
        sys.exit(1)
    print(f"\n总计: {len(texts)} 个文件, {total_chars} 字符")

    # ── Queries ───────────────────────────────────────────────────────
    queries = args.query if args.query else _find_queries(file_paths)
    print(f"\n测试查询 ({len(queries)}):")
    for q in queries:
        print(f"  · {q}")

    # ── LLM config ────────────────────────────────────────────────────
    llm_config = None
    if args.llm:
        # Default to ollama if no endpoint/key provided
        style = args.api_style
        endpoint = args.api_endpoint
        key = args.api_key
        model = args.model or "qwen2.5:7b"
        if not endpoint and not key:
            style = "ollama"
            endpoint = "http://localhost:11434"
            model = model or "qwen2.5:7b"
        llm_config = {
            "api_style": style,
            "api_endpoint": endpoint,
            "api_key": key,
            "model": model,
        }
        print(f"\nLLM: {style} / {model} @ {endpoint or 'default'}")
    else:
        print("\n(未启用 LLM 评估，仅测试检索指标。加 --llm 启用)")

    # ── Run ───────────────────────────────────────────────────────────
    results = run_size_benchmark(
        texts, queries,
        chunk_sizes=args.sizes,
        llm_config=llm_config,
    )

    # ── Print ─────────────────────────────────────────────────────────
    print_table(results, has_llm=bool(llm_config))

    # ── JSON output ───────────────────────────────────────────────────
    json_path = Path("result") / "chunk_size_benchmark.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in results:
        d = {
            "chunk_size": r.chunk_size,
            "chunk_count": r.chunk_count,
            "avg_length": round(r.avg_length, 1),
            "recall_at_3": round(r.recall_at_3, 4),
            "recall_at_5": round(r.recall_at_5, 4),
            "mrr": round(r.mrr, 4),
            "retrieval_ms": round(r.retrieval_ms, 1),
            "embed_ms": round(r.embed_ms, 1),
        }
        if llm_config:
            d.update({
                "llm_relevance": round(r.llm_relevance, 2),
                "llm_completeness": round(r.llm_completeness, 2),
                "llm_factual": round(r.llm_factual, 2),
                "llm_latency_s": round(r.llm_latency_s, 1),
                "answers": r.answers,
            })
        payload.append(d)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {json_path}")


if __name__ == "__main__":
    main()
