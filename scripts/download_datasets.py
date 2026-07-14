"""Download RAG evaluation datasets from HuggingFace.

Usage:
  python scripts/download_datasets.py --output ./datasets --subsets 1000
"""

import argparse
import json
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


DATASETS = {
    "nq_open": {
        "name": "nq_open",
        "desc": "Natural Questions Open — Google queries + Wikipedia answers (English)",
        "splits": {"train": 500},
        "query_field": "question",
        "answer_field": "answer",
        "doc_field": None,
    },
    "mlqa_zh": {
        "name": "mlqa",
        "config": "mlqa.zh.zh",
        "desc": "MLQA Chinese subset — cross-lingual QA (Chinese)",
        "splits": {"test": 500},
        "query_field": "question",
        "answer_field": "answers",
        "doc_field": "context",
    },
    "clue_csl": {
        "name": "clue",
        "config": "csl",
        "desc": "CLUE CSL — Chinese scientific literature abstracts + keywords",
        "splits": {"train": 1000},
        "query_field": "keyword",
        "answer_field": "abst",
        "doc_field": "title",
    },
    "multi_news": {
        "name": "multi_news",
        "desc": "Multi-News — multi-document summarization (English)",
        "splits": {"test": 300},
        "query_field": "summary",
        "answer_field": "document",
        "doc_field": None,
    },
    "cnn_dailymail": {
        "name": "cnn_dailymail",
        "config": "3.0.0",
        "desc": "CNN/Daily Mail — news summarization (English)",
        "splits": {"test": 300},
        "query_field": "highlights",
        "answer_field": "article",
        "doc_field": None,
    },
}


def download_dataset(cfg: dict, max_samples: int, output_dir: Path):
    """Download one dataset and save as JSON lines."""
    from datasets import load_dataset

    ds_name = cfg["name"]
    ds_config = cfg.get("config")
    desc = cfg["desc"]
    out_file = output_dir / f"{ds_name}.jsonl"

    print(f"\n[{ds_name}] {desc}")

    all_records = []
    for split_name, limit in cfg["splits"].items():
        n = min(limit, max_samples)
        print(f"  Loading {split_name} (max {n})...", end=" ", flush=True)
        try:
            kwargs = {"path": ds_name, "split": f"{split_name}[:{n}]"}
            if ds_config:
                kwargs["name"] = ds_config
            ds = load_dataset(**kwargs)
        except Exception as e:
            print(f"FAIL: {e}")
            continue

        qf = cfg["query_field"]
        af = cfg["answer_field"]
        df = cfg.get("doc_field")

        records = []
        for item in ds:
            rec = {}
            # Query
            if qf in item:
                qval = item[qf]
                if isinstance(qval, list):
                    qval = qval[0] if qval else ""
                elif isinstance(qval, dict):
                    qval = str(list(qval.values())[0]) if qval else ""
                rec["query"] = str(qval).strip()
            else:
                rec["query"] = ""

            # Answer
            if af in item:
                aval = item[af]
                if isinstance(aval, list):
                    aval = " ".join(str(a) for a in aval[:3])
                rec["answer"] = str(aval).strip()
            else:
                rec["answer"] = ""

            # Document/context
            if df and df in item:
                dval = item[df]
                if isinstance(dval, list):
                    dval = " ".join(str(d) for d in dval[:5])
                rec["document"] = str(dval).strip()
            else:
                rec["document"] = ""

            if rec["query"] and rec["answer"]:
                records.append(rec)

        print(f"{len(records)} records")
        all_records.extend(records)

    if all_records:
        with open(out_file, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(all_records)} records -> {out_file}")
    else:
        print(f"  No records downloaded for {ds_name}")

    return len(all_records)


def download_dureader(output_dir: Path):
    """DuReader — Chinese reading comprehension from Baidu.

    Downloads from HuggingFace mirror (dureader_robust).
    """
    from datasets import load_dataset

    out_file = output_dir / "dureader.jsonl"
    print(f"\n[dureader] DuReader — Chinese reading comprehension (Baidu)")

    records = []
    for split, limit in [("train", 500), ("validation", 200)]:
        n = min(limit, 1000)
        print(f"  Loading {split} (max {n})...", end=" ", flush=True)
        try:
            ds = load_dataset("dureader_robust", split=f"{split}[:{n}]")
        except Exception as e:
            print(f"FAIL: {e}")
            # Try alternate name
            try:
                ds = load_dataset("PaddlePaddle/dureader_robust", split=f"{split}[:{n}]")
            except Exception as e2:
                print(f"FAIL (alternate): {e2}")
                continue

        for item in ds:
            rec = {
                "query": str(item.get("question") or "").strip(),
                "answer": str(item.get("answers") or [""])[0] if item.get("answers") else "",
                "document": str(item.get("context") or "")[:2000],
            }
            if rec["query"] and rec["answer"]:
                records.append(rec)
        print(f"{len(records)} records so far")

    if records:
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(records)} records -> {out_file}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Download RAG evaluation datasets")
    parser.add_argument("--output", default="./datasets", help="Output directory")
    parser.add_argument("--subsets", type=int, default=500,
                        help="Max samples per split")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Dataset names to skip (e.g. --skip cnn_dailymail multi_news)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RAG Evaluation Dataset Downloader")
    print("=" * 60)
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Max samples per split: {args.subsets}")
    print()

    skip_set = set(args.skip or [])

    total = 0
    for name, cfg in DATASETS.items():
        if name in skip_set:
            print(f"\n[{name}] SKIPPED")
            continue
        n = download_dataset(cfg, args.subsets, output_dir)
        total += n

    # DuReader (separate download logic)
    if "dureader" not in skip_set:
        try:
            n = download_dureader(output_dir)
            total += n
        except Exception as e:
            print(f"\n[dureader] SKIPPED — {e}")

    print(f"\n{'='*60}")
    print(f"  Done — {total} total records across all datasets")
    print(f"  Files in: {output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()