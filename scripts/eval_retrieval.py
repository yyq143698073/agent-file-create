"""Retrieval evaluation script — measures Recall@k, MRR, NDCG@k.

Usage:
  python scripts/eval_retrieval.py --kb default --test-data tests/retrieval_queries.json

Test data format (retrieval_queries.json):
[
  {
    "query": "风险管理政策是什么？",
    "relevant_doc_ids": ["doc_a", "doc_c"],
    "relevant_chunk_ids": ["doc_a:3", "doc_a:4"]   // optional, for fine-grained eval
  },
  ...
]

Output:
  Recall@5, Recall@10, MRR, NDCG@5, NDCG@10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


def load_test_data(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test data file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Test data must be a JSON array")
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query") or "").strip()
        if not q:
            continue
        rel_docs = item.get("relevant_doc_ids", [])
        rel_chunks = item.get("relevant_chunk_ids", [])
        if not isinstance(rel_docs, list):
            rel_docs = []
        if not isinstance(rel_chunks, list):
            rel_chunks = []
        out.append({
            "query": q,
            "relevant_doc_ids": [str(d) for d in rel_docs if d],
            "relevant_chunk_ids": [str(c) for c in rel_chunks if c],
        })
    return out


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    rel = set(relevant_ids)
    return len(top & rel) / len(rel)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    rel = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """NDCG where relevant=1, non-relevant=0 (binary relevance)."""
    if not relevant_ids:
        return 0.0
    rel = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in rel:
            dcg += 1.0 / __import__("math").log2(i + 2)  # i+2 because i is 0-indexed
    # Ideal DCG: all relevant at top
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / __import__("math").log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def run_eval(
    kb_name: str = "default",
    test_data_path: str = "",
    top_k: int = 10,
) -> dict:
    from agent_file_create.rag.kb import KnowledgeBase

    items = load_test_data(test_data_path)
    if not items:
        return {"error": "no_valid_queries"}

    kb = KnowledgeBase()
    metrics: dict[str, list[float]] = {
        "recall@5": [], "recall@10": [],
        "mrr": [],
        "ndcg@5": [], "ndcg@10": [],
    }
    per_query: list[dict] = []

    for item in items:
        q = item["query"]
        rel_docs = item["relevant_doc_ids"]
        rel_chunks = item["relevant_chunk_ids"]

        try:
            hits = kb.search(kb=kb_name, query=q, top_k=top_k)
        except Exception as exc:
            per_query.append({"query": q, "error": str(exc)[:200]})
            continue

        # Evaluate by doc_id
        retrieved_doc_ids = [str(h.doc_id or "") for h in hits]
        # Evaluate by chunk_id (if fine-grained labels provided)
        retrieved_chunk_ids = [str(h.chunk_id or "") for h in hits]

        r5_doc = recall_at_k(retrieved_doc_ids, rel_docs, 5)
        r10_doc = recall_at_k(retrieved_doc_ids, rel_docs, 10)
        m = mrr(retrieved_doc_ids, rel_docs)
        n5_doc = ndcg_at_k(retrieved_doc_ids, rel_docs, 5)
        n10_doc = ndcg_at_k(retrieved_doc_ids, rel_docs, 10)

        # If chunk-level labels exist, also report chunk-level
        chunk_metrics = {}
        if rel_chunks:
            chunk_metrics = {
                "recall@5_chunk": recall_at_k(retrieved_chunk_ids, rel_chunks, 5),
                "mrr_chunk": mrr(retrieved_chunk_ids, rel_chunks),
            }

        metrics["recall@5"].append(r5_doc)
        metrics["recall@10"].append(r10_doc)
        metrics["mrr"].append(m)
        metrics["ndcg@5"].append(n5_doc)
        metrics["ndcg@10"].append(n10_doc)

        per_query.append({
            "query": q,
            "recall@5": round(r5_doc, 3),
            "recall@10": round(r10_doc, 3),
            "mrr": round(m, 3),
            "ndcg@5": round(n5_doc, 3),
            "top_docs": retrieved_doc_ids[:5],
            **chunk_metrics,
        })

    n = len(metrics["recall@5"]) or 1
    summary = {
        "num_queries": len(items),
        "num_evaluated": n,
        "avg_recall@5": round(sum(metrics["recall@5"]) / n, 4),
        "avg_recall@10": round(sum(metrics["recall@10"]) / n, 4),
        "avg_mrr": round(sum(metrics["mrr"]) / n, 4),
        "avg_ndcg@5": round(sum(metrics["ndcg@5"]) / n, 4),
        "avg_ndcg@10": round(sum(metrics["ndcg@10"]) / n, 4),
    }
    return {"summary": summary, "per_query": per_query}


def main():
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval quality")
    parser.add_argument("--kb", default="default", help="Knowledge base name")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k (default 10)")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    result = run_eval(kb_name=args.kb, test_data_path=args.test_data, top_k=args.top_k)
    out_json = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(out_json, encoding="utf-8")
        print(f"Results written to {args.output}")

    print(out_json)


if __name__ == "__main__":
    main()
