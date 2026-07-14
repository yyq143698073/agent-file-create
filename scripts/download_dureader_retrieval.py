"""Download DuReader-Retrieval — Chinese passage retrieval benchmark.

Downloads a manageable subset and converts to our eval format.

Usage:
  python scripts/download_dureader_retrieval.py --output ./datasets --max-docs 500
"""

import argparse
import json
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


def download_via_api(output_dir: Path, max_docs: int):
    """Download DuReader retrieval via HuggingFace API."""
    # Avoid name collision with project's ./datasets/ directory
    import importlib.util
    _spec = importlib.util.find_spec("datasets")
    if _spec and _spec.origin and "site-packages" in _spec.origin:
        from datasets import load_dataset
    else:
        raise ImportError(
            "Cannot import 'datasets' library — project's ./datasets/ directory is shadowing it. "
            "Rename the directory or install: pip install datasets"
        )

    print("Downloading DuReader retrieval via HuggingFace...")

    # Try the zyznull mirror first
    dataset_names = [
        "zyznull/dureader-retrieval-ranking",
        "luozhouyang/dureader",
    ]

    all_docs = {}    # doc_id → text
    all_queries = []  # {query, relevant_doc_ids}

    for ds_name in dataset_names:
        print(f"\n  Trying: {ds_name}")
        try:
            if "ranking" in ds_name:
                ds = load_dataset(ds_name, split="train")
            else:
                ds = load_dataset(ds_name, "robust", split="test")
        except Exception as e:
            print(f"  FAIL: {e}")
            continue

        print(f"  Loaded {len(ds)} records")

        for item in ds:
            # Extract query
            query = str(item.get("question") or item.get("query") or "").strip()
            if not query:
                continue

            # Extract passages/documents
            passages = item.get("passages") or item.get("context") or []
            if isinstance(passages, dict):
                passages = [passages]
            if not isinstance(passages, list):
                continue

            relevant_ids = []
            for pi, p in enumerate(passages[:20]):  # max 20 passages per query
                if isinstance(p, dict):
                    text = str(p.get("passage_text") or p.get("text") or p.get("content") or "").strip()
                    is_selected = p.get("is_selected", 0)
                elif isinstance(p, str):
                    text = p.strip()
                    is_selected = 1  # assume all are relevant
                else:
                    continue

                if not text:
                    continue

                doc_id = f"dureader_doc_{abs(hash(text)) % 10**8:08d}"
                if doc_id not in all_docs:
                    all_docs[doc_id] = text

                if is_selected:
                    relevant_ids.append(doc_id)

            if query and relevant_ids:
                all_queries.append({
                    "query": query,
                    "relevant_doc_ids": relevant_ids[:5],
                })

        if all_queries:
            break  # got data, stop trying

    # Limit document count
    if len(all_docs) > max_docs:
        # Keep docs that are referenced by at least one query
        referenced = set()
        for q in all_queries:
            referenced.update(q["relevant_doc_ids"])
        all_docs = {k: v for k, v in all_docs.items() if k in referenced}
        all_docs = dict(list(all_docs.items())[:max_docs])

    print(f"\n  Result: {len(all_queries)} queries, {len(all_docs)} documents")

    # Save documents
    docs_file = output_dir / "dureader_docs.jsonl"
    with open(docs_file, "w", encoding="utf-8") as f:
        for doc_id, text in all_docs.items():
            f.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")

    # Save queries
    queries_file = output_dir / "dureader_queries.json"
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {docs_file} ({len(all_docs)} docs)")
    print(f"  Saved: {queries_file} ({len(all_queries)} queries)")

    return len(all_queries), len(all_docs)


def download_via_file(output_dir: Path):
    """Fallback: download pre-converted files from GitHub mirror."""
    import urllib.request
    import zipfile
    import tempfile

    urls = [
        "https://github.com/baidu/DuReader/archive/refs/heads/master.zip",
    ]

    print("Trying direct download from GitHub...")
    for url in urls:
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                with zipfile.ZipFile(tmp.name, "r") as zf:
                    zf.extractall(output_dir)
                print(f"  Downloaded and extracted: {url}")
                return 0, 0
        except Exception as e:
            print(f"  FAIL: {e}")
    return 0, 0


def main():
    parser = argparse.ArgumentParser(description="Download DuReader-Retrieval")
    parser.add_argument("--output", default="./datasets", help="Output directory")
    parser.add_argument("--max-docs", type=int, default=500,
                        help="Max documents to keep")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DuReader-Retrieval Downloader")
    print("  Chinese passage retrieval benchmark (Baidu)")
    print("=" * 60)

    n_queries, n_docs = download_via_api(output_dir, args.max_docs)
    if n_queries == 0:
        download_via_file(output_dir)

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()