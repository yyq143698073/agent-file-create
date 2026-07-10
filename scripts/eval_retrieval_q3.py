"""Q3 retrieval evaluation using BEIR NFCorpus via ir_datasets.

Usage:
  python scripts/eval_retrieval_q3.py --sample 100 --output result/q3_eval_v2_baseline.json
  python scripts/eval_retrieval_q3.py --sample 100 --adaptive-weights --ablate rrf_adaptive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_kb(dataset, kb_name: str, max_docs: int) -> Any:
    """Ingest BEIR docs into a KnowledgeBase via temp files and return it."""
    import tempfile
    from agent_file_create.rag.kb import KnowledgeBase

    kb = KnowledgeBase()
    kb_name_norm = kb_name.strip()

    print(f"  Ingesting up to {max_docs} docs into KB '{kb_name_norm}' ...")
    docs_iter = dataset.docs_iter()
    count = 0
    t0 = time.time()

    tmpdir = Path(tempfile.mkdtemp(prefix="q3_eval_"))

    for doc in docs_iter:
        title = str(getattr(doc, "title", "") or "")
        body = str(getattr(doc, "text", "") or "")
        content = f"{title}\n\n{body}" if title else body
        if not content.strip():
            continue

        did = str(getattr(doc, "doc_id", f"nfc_{count}") or f"nfc_{count}")
        # Write to temp file
        tmpfile = tmpdir / f"{did}.txt"
        tmpfile.write_text(content, encoding="utf-8")

        try:
            kb.ingest_file(
                kb=kb_name_norm,
                file_path=str(tmpfile),
                doc_id=did,
                title=title or did,
                source="beir/nfcorpus",
            )
        except Exception as e:
            print(f"  WARN: ingest failed for {did}: {str(e)[:100]}")

        count += 1
        if count >= max_docs:
            break
        if count % 20 == 0:
            print(f"    {count}/{max_docs} ...")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"  Done: {count} docs in {elapsed:.0f}s")
    return kb


def _evaluate_queries(
    kb,
    kb_name: str,
    queries: list[dict],
    qrels: dict[str, set[str]],
    top_k: int,
    adaptive_weights: bool,
) -> dict:
    """Run queries and compute Recall / Precision / MRR."""
    recall_scores: list[float] = []
    precision_scores: list[float] = []
    mrr_scores: list[float] = []
    details: list[dict] = []
    seconds_list: list[float] = []

    for idx, q_item in enumerate(queries):
        qid = str(q_item["qid"])
        qtext = str(q_item["text"] or "").strip()
        if not qtext:
            continue

        relevant = qrels.get(qid, set())
        if not relevant:
            continue

        t0 = time.time()
        if adaptive_weights:
            hits = kb.search_adaptive(kb=kb_name, query=qtext, top_k=top_k)
        else:
            hits = kb.search(kb=kb_name, query=qtext, top_k=top_k)
        elapsed = time.time() - t0
        seconds_list.append(elapsed)

        # Recall@K
        hit_doc_ids = {str(h.doc_id or "") for h in hits}
        recalled = len(hit_doc_ids & relevant)
        recall = recalled / max(len(relevant), 1)
        recall_scores.append(recall)

        # Precision@K
        precision = recalled / max(len(hit_doc_ids), 1) if hit_doc_ids else 0.0
        precision_scores.append(precision)

        # MRR
        mrr = 0.0
        for rank, h in enumerate(hits, start=1):
            if str(h.doc_id or "") in relevant:
                mrr = 1.0 / float(rank)
                break
        mrr_scores.append(mrr)

        details.append({
            "qid": qid,
            "query": qtext[:120],
            "relevant_count": len(relevant),
            "hit_count": len(hits),
            "recalled": recalled,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "mrr": round(mrr, 4),
            "seconds": round(elapsed, 4),
        })

        if (idx + 1) % 20 == 0:
            r_mean = sum(recall_scores) / len(recall_scores)
            print(f"  [{idx + 1}/{len(queries)}] Recall@{top_k}={r_mean:.3f}")

    import statistics

    def _ms(vals):
        if not vals:
            return 0.0, 0.0
        if len(vals) == 1:
            return round(vals[0], 4), 0.0
        return round(statistics.mean(vals), 4), round(statistics.pstdev(vals), 4)

    r_mean, r_std = _ms(recall_scores)
    p_mean, p_std = _ms(precision_scores)
    m_mean, m_std = _ms(mrr_scores)
    avg_s = round(statistics.mean(seconds_list), 4) if seconds_list else 0.0

    return {
        "sample_count": len(details),
        "top_k": top_k,
        "recall": r_mean,
        "recall_std": r_std,
        "precision": p_mean,
        "precision_std": p_std,
        "mrr": m_mean,
        "mrr_std": m_std,
        "avg_seconds": avg_s,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q3 retrieval evaluation on BEIR NFCorpus")
    parser.add_argument("--sample", type=int, default=100, help="Number of docs to ingest (default 100)")
    parser.add_argument("--query-sample", type=int, default=50, help="Number of queries to evaluate (default 50)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--adaptive-weights", action="store_true", default=True, help="Enable adaptive RRF weights/k (default on)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive RRF weights/k")
    parser.add_argument("--ablate", default="", help="Comma-separated ablation flags")
    parser.add_argument("--output", default="result/q3_eval_v2_baseline.json")
    args = parser.parse_args()

    # Set env for ablation
    ablate_set = {x.strip().lower() for x in str(args.ablate or "").split(",") if x.strip()}
    for flag in ("rrf_adaptive", "dedup", "body_query_expand"):
        os.environ[f"Q3_ENABLE_{flag.upper()}"] = "false" if flag in ablate_set else "true"

    print("=" * 50)
    print("Q3 Retrieval Evaluation — BEIR NFCorpus")
    print(f"  Docs: {args.sample}, Queries: {args.query_sample}, Top-K: {args.top_k}")
    print(f"  Adaptive: {not args.no_adaptive}, Ablate: {sorted(ablate_set) or 'none'}")
    print()

    # 1. Load dataset
    print("[1/4] Loading NFCorpus via ir_datasets ...")
    import ir_datasets
    ds = ir_datasets.load("beir/nfcorpus")
    print(f"  Dataset: {ds.docs_count()} docs total")

    # 2. Build KB
    print(f"[2/4] Building KB with {args.sample} docs ...")
    kb = _build_kb(ds, "q3_eval_nfcorpus", args.sample)

    # 3. Load queries and qrels
    print("[3/4] Loading queries and qrels ...")
    # Load qrels from zip file FIRST (so we can filter queries to only those with relevance judgments)
    qrels: dict[str, set[str]] = {}
    import zipfile as _zf
    _cache = os.path.expanduser("~/.ir_datasets/beir/nfcorpus")
    with _zf.ZipFile(_cache + "/source.zip", "r") as _z:
        _raw = _z.read("nfcorpus/qrels/test.tsv").decode("utf-8")
        for _line in _raw.strip().split("\n")[1:]:
            _parts = _line.split("\t")
            if len(_parts) >= 3:
                _qid, _did = _parts[0].strip(), _parts[1].strip()
                _score = int(_parts[2].strip() or "0")
                if _score > 0:
                    if _qid not in qrels:
                        qrels[_qid] = set()
                    qrels[_qid].add(_did)
    print(f"  Qrels loaded: {len(qrels)} queries with relevance judgments")

    # Load queries — only those with qrels
    query_sample = args.query_sample
    queries: list[dict] = []
    for q in ds.queries_iter():
        qid = str(getattr(q, "query_id", ""))
        if qid in qrels:
            queries.append({"qid": qid, "text": str(getattr(q, "text", "") or "")})
            if len(queries) >= query_sample:
                break

    queries_with_qrels = queries  # all have qrels
    print(f"  Queries with relevance judgments: {len(queries_with_qrels)}")

    # 4. Evaluate
    print(f"[4/4] Running evaluation ({len(queries_with_qrels)} queries) ...")
    t0 = time.time()
    use_adaptive = not args.no_adaptive
    report = _evaluate_queries(
        kb, "q3_eval_nfcorpus",
        queries_with_qrels, qrels,
        top_k=args.top_k,
        adaptive_weights=use_adaptive,
    )
    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 50)
    print(f"Recall@{args.top_k}:   {report['recall']:.4f} ± {report['recall_std']:.4f}")
    print(f"Precision@{args.top_k}: {report['precision']:.4f} ± {report['precision_std']:.4f}")
    print(f"MRR@{args.top_k}:       {report['mrr']:.4f} ± {report['mrr_std']:.4f}")
    print(f"Avg time/query:      {report['avg_seconds']:.4f}s")
    print(f"Total time:          {elapsed:.0f}s")
    print()

    # Save
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset": "beir/nfcorpus",
            "docs_ingested": args.sample,
            "queries_evaluated": len(queries_with_qrels),
            "top_k": args.top_k,
            "adaptive_weights": use_adaptive,
            "ablation": sorted(ablate_set),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_seconds": round(elapsed, 2),
        },
        "summary": {
            k: v for k, v in report.items() if k != "details"
        },
        "details": report.get("details", []),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
