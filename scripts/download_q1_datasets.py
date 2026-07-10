"""Download and prepare Q1 evaluation datasets.

Downloads fixed samples from each dataset and saves images + ground truth locally.
This ensures reproducible evaluation without depending on HF connectivity at eval time.

Datasets:
  - DocVQA     : nielsr/docvqa_1200_examples  (document understanding, 200 test)
  - FUNSD      : jinho8345/funsd               (form key-value extraction, 50 test)
  - PubTabNet  : apoidea/pubtabnet-html        (table structure, 9115 validation)
  - SROIE      : jsdnrs/ICDAR2019-SROIE        (receipt info extraction, 361 test)
  - CTW        : manual download from https://ctwdataset.github.io/ (Chinese OCR)

Usage:
  python scripts/download_q1_datasets.py                          # default counts
  python scripts/download_q1_datasets.py --pubtabnet 200 --sroie 100
  python scripts/download_q1_datasets.py --skip ctw               # skip CTW note
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "result" / "q1_eval_v2" / "samples"

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "docvqa": {
        "name": "DocVQA",
        "hf_id": "nielsr/docvqa_1200_examples",
        "split": "test",
        "default_count": 100,
        "max_count": 200,
        "description": "Document Visual QA — answer hit rate via ANLS",
    },
    "funsd": {
        "name": "FUNSD",
        "hf_id": "jinho8345/funsd",
        "split": "test",
        "default_count": 50,
        "max_count": 50,
        "description": "Form Understanding in Noisy Scanned Documents — KV-pair F1",
    },
    "pubtabnet": {
        "name": "PubTabNet",
        "hf_id": "apoidea/pubtabnet-html",
        "split": "validation",
        "default_count": 100,
        "max_count": 9115,
        "description": "Table structure recognition — TEDS (Tree-Edit-Distance Similarity)",
    },
    "sroie": {
        "name": "SROIE",
        "hf_id": "jsdnrs/ICDAR2019-SROIE",
        "split": "test",
        "default_count": 100,
        "max_count": 361,
        "description": "Scanned Receipts OCR and Information Extraction — KV-pair F1",
    },
    "ctw": {
        "name": "CTW",
        "hf_id": None,  # manual download
        "split": None,
        "default_count": 0,
        "max_count": 0,
        "description": "Chinese Text in the Wild — Character Error Rate. "
        "Requires manual download from https://ctwdataset.github.io/",
    },
}


def _save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(str(path))
    else:
        raise TypeError(f"unsupported image type: {type(image)}")


def download_docvqa(count: int, output_dir: Path) -> list[dict]:
    """Download DocVQA samples: image + question + answers."""
    from datasets import load_dataset

    ds = load_dataset("nielsr/docvqa_1200_examples", split=f"test[:{count}]")
    img_dir = output_dir / "docvqa" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for idx, item in enumerate(ds):
        fname = f"docvqa_{idx:04d}.png"
        _save_image(item["image"], img_dir / fname)

        answers = [str(a) for a in item.get("answers", []) if str(a).strip()]
        record = {
            "id": item.get("id"),
            "file": fname,
            "question": (item.get("query") or {}).get("en", ""),
            "answers": answers,
        }
        samples.append(record)
        if (idx + 1) % 20 == 0:
            print(f"  [DocVQA] {idx + 1}/{count}")

    # Save ground truth
    gt_path = output_dir / "docvqa" / "ground_truth.json"
    gt_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [DocVQA] saved {len(samples)} samples → {gt_path}")
    return samples


def download_funsd(count: int, output_dir: Path) -> list[dict]:
    """Download FUNSD samples: image + key-value pairs."""
    from datasets import load_dataset

    ds = load_dataset("jinho8345/funsd", split=f"test[:{count}]")
    img_dir = output_dir / "funsd" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for idx, item in enumerate(ds):
        fname = f"funsd_{idx:04d}.png"
        _save_image(item["img"], img_dir / fname)

        # Extract key-value pairs from FUNSD annotations
        labels = item.get("labels", []) or []
        words = item.get("words", []) or []
        pairs: list[dict[str, str]] = []
        current_q = ""
        for label, word_group in zip(labels, words):
            text = " ".join(
                str((token or {}).get("text") or "").strip()
                for token in (word_group or [])
                if str((token or {}).get("text") or "").strip()
            ).strip()
            if not text:
                continue
            if label == "question":
                current_q = text
            elif label == "answer" and current_q:
                pairs.append({"field": current_q, "value": text})
                current_q = ""
            elif label == "other" and current_q:
                # "other" label following a question may contain multi-line values
                pass

        record = {
            "filename": item.get("filename"),
            "file": fname,
            "pairs": pairs,
            "pair_count": len(pairs),
        }
        samples.append(record)
        if (idx + 1) % 20 == 0:
            print(f"  [FUNSD] {idx + 1}/{count}")

    gt_path = output_dir / "funsd" / "ground_truth.json"
    gt_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [FUNSD] saved {len(samples)} samples → {gt_path}")
    return samples


def download_pubtabnet(count: int, output_dir: Path) -> list[dict]:
    """Download PubTabNet samples: image + HTML table structure."""
    from datasets import load_dataset

    ds = load_dataset("apoidea/pubtabnet-html", split=f"validation[:{count}]")
    img_dir = output_dir / "pubtabnet" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for idx, item in enumerate(ds):
        fname = f"pubtabnet_{idx:04d}.png"
        _save_image(item["image"], img_dir / fname)

        html_table = str(item.get("html_table") or "")
        record = {
            "imgid": item.get("imgid"),
            "split": item.get("split"),
            "file": fname,
            "html_table": html_table,
        }
        samples.append(record)
        if (idx + 1) % 50 == 0:
            print(f"  [PubTabNet] {idx + 1}/{count}")

    gt_path = output_dir / "pubtabnet" / "ground_truth.json"
    gt_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [PubTabNet] saved {len(samples)} samples → {gt_path}")
    return samples


def download_sroie(count: int, output_dir: Path) -> list[dict]:
    """Download SROIE samples: receipt image + entity annotations."""
    from datasets import load_dataset

    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split=f"test[:{count}]")
    img_dir = output_dir / "sroie" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for idx, item in enumerate(ds):
        fname = f"sroie_{idx:04d}.jpg"
        _save_image(item["image"], img_dir / fname)

        entities = item.get("entities") or {}
        record = {
            "key": item.get("key"),
            "file": fname,
            "entities": {
                "company": str(entities.get("company") or ""),
                "date": str(entities.get("date") or ""),
                "address": str(entities.get("address") or ""),
                "total": str(entities.get("total") or ""),
            },
            "words": item.get("words") or [],
            "image_size": item.get("image_size"),
        }
        samples.append(record)
        if (idx + 1) % 50 == 0:
            print(f"  [SROIE] {idx + 1}/{count}")

    gt_path = output_dir / "sroie" / "ground_truth.json"
    gt_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [SROIE] saved {len(samples)} samples → {gt_path}")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare Q1 evaluation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_q1_datasets.py\n"
            "  python scripts/download_q1_datasets.py --pubtabnet 200 --sroie 100\n"
            "  python scripts/download_q1_datasets.py --docvqa 0 --skip funsd  # only download selected\n"
        ),
    )
    parser.add_argument("--docvqa", type=int, default=None, help=f"DocVQA sample count (default: 100, max: 200)")
    parser.add_argument("--funsd", type=int, default=None, help=f"FUNSD sample count (default: 50, max: 50)")
    parser.add_argument("--pubtabnet", type=int, default=None, help=f"PubTabNet sample count (default: 100, max: 9115)")
    parser.add_argument("--sroie", type=int, default=None, help=f"SROIE sample count (default: 100, max: 361)")
    parser.add_argument("--ctw", type=int, default=0, help="CTW count (requires manual download from https://ctwdataset.github.io/)")
    parser.add_argument("--skip", nargs="*", default=[], help="Datasets to skip (e.g. --skip ctw funsd)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    skip_set = set(args.skip or [])

    # Resolve counts
    counts: dict[str, int] = {}
    for key, spec in DATASET_SPECS.items():
        if key in skip_set:
            counts[key] = 0
            continue
        user_val = getattr(args, key, None)
        if user_val is not None:
            counts[key] = max(0, min(user_val, spec["max_count"]))
        else:
            counts[key] = min(spec["default_count"], spec["max_count"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Q1 Dataset Download")
    print(f"Output: {output_dir}")
    print()

    downloaders = {
        "docvqa": download_docvqa,
        "funsd": download_funsd,
        "pubtabnet": download_pubtabnet,
        "sroie": download_sroie,
    }

    manifest: dict[str, Any] = {"datasets": {}}

    for key, count in counts.items():
        spec = DATASET_SPECS[key]
        print(f"[{spec['name']}] {spec['description']}")

        if key == "ctw":
            if count > 0:
                print(f"  ⚠ CTW requires manual download from https://ctwdataset.github.io/")
                print(f"  After download, place images in: {output_dir / 'ctw' / 'images' / ''}")
                print(f"  Expected format: one .jpg/.png per image, with ground truth labels")
            manifest["datasets"][key] = {
                "name": spec["name"],
                "count": 0,
                "status": "manual_download_required",
                "url": "https://ctwdataset.github.io/",
            }
            print()
            continue

        if count == 0:
            print("  → skipped")
            manifest["datasets"][key] = {"name": spec["name"], "count": 0, "status": "skipped"}
            print()
            continue

        print(f"  Downloading {count} samples from {spec['hf_id']} ({spec['split']})...")
        try:
            samples = downloaders[key](count, output_dir)
            manifest["datasets"][key] = {
                "name": spec["name"],
                "hf_id": spec["hf_id"],
                "split": spec["split"],
                "count": len(samples),
                "status": "ok",
                "ground_truth": str(output_dir / key / "ground_truth.json"),
            }
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            manifest["datasets"][key] = {
                "name": spec["name"],
                "hf_id": spec["hf_id"],
                "count": 0,
                "status": f"failed: {str(e)[:200]}",
            }
        print()

    # Write manifest
    total = sum(d.get("count", 0) for d in manifest["datasets"].values())
    manifest["total_samples"] = total
    manifest["output_dir"] = str(output_dir)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"Download complete. Total samples: {total}")
    print(f"Manifest: {manifest_path}")
    print()
    for key, info in manifest["datasets"].items():
        status = info["status"]
        count = info["count"]
        marker = "[OK]" if status == "ok" else "[FAIL]" if "failed" in status else "[WARN]"
        print(f"  {marker} {info['name']}: {count} samples ({status})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
