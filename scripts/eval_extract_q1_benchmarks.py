from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any


def _norm_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _contains_answer(summary_text: str, answers: list[str]) -> bool:
    hay = _norm_text(summary_text)
    if not hay:
        return False
    for ans in answers:
        cand = _norm_text(ans)
        if cand and cand in hay:
            return True
    return False


def _save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(path)
        return
    if isinstance(image, dict) and image.get("bytes") is not None:
        path.write_bytes(image["bytes"])
        return
    raise TypeError("unsupported image type")


def _format_funsd_as_text(item: dict) -> str:
    parts: list[str] = []
    for label, word_group in zip(item.get("labels", []), item.get("words", [])):
        tokens = []
        for token in word_group or []:
            text = str((token or {}).get("text") or "").strip()
            if text:
                tokens.append(text)
        if tokens:
            parts.append(f"[{label}] " + " ".join(tokens))
    return "\n".join(parts)


def _funsd_key_value_pairs(item: dict) -> list[tuple[str, str]]:
    labels = item.get("labels", []) or []
    groups = item.get("words", []) or []
    pairs: list[tuple[str, str]] = []
    current_q = ""
    for label, word_group in zip(labels, groups):
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
            pairs.append((current_q, text))
            current_q = ""
    return pairs


def _tabfact_table_markdown(table_text: str) -> str:
    rows = []
    for raw in str(table_text or "").split("\n"):
        cols = [c.strip() for c in raw.split("#") if c.strip()]
        if cols:
            rows.append(cols)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    head = rows[0]
    body = rows[1:] or [[""] * width]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def evaluate_docvqa(root: Path, sample_count: int, work_dir: Path) -> dict:
    if sample_count <= 0:
        return {
            "dataset": "DocVQA",
            "metric": "answer_hit_rate",
            "sample_count": 0,
            "score": 0.0,
            "target": 0.85,
            "cases": [],
        }

    from datasets import load_dataset

    from agent_file_create.document.extractor import extract_from_file

    ds = load_dataset("nielsr/docvqa_1200_examples", split=f"test[:{sample_count}]")
    cases = []
    hit = 0
    for idx, item in enumerate(ds):
        print(f"[DocVQA] {idx + 1}/{len(ds)}")
        img_path = work_dir / "docvqa" / f"docvqa_{idx}.png"
        _save_image(item["image"], img_path)
        result = extract_from_file(str(img_path), preprocess=True)
        answers = [str(a) for a in item.get("answers", []) if str(a).strip()]
        joined = "\n".join(
            [
                str(result.get("title") or ""),
                str(result.get("summary") or ""),
                " ".join(str(x) for x in (result.get("key_points") or [])),
                str(result.get("_ocr_text") or ""),
                json.dumps(result.get("data", ""), ensure_ascii=False),
            ]
        )
        ok = _contains_answer(joined, answers)
        hit += int(ok)
        cases.append(
            {
                "id": item.get("id"),
                "question": ((item.get("query") or {}).get("en") or ""),
                "answers": answers[:3],
                "pred_title": result.get("title", ""),
                "pred_summary": str(result.get("summary") or "")[:180],
                "matched": ok,
            }
        )
    return {
        "dataset": "DocVQA",
        "metric": "answer_hit_rate",
        "sample_count": len(cases),
        "score": round(hit / float(len(cases) or 1), 4),
        "target": 0.85,
        "cases": cases,
    }


def evaluate_funsd(root: Path, sample_count: int, work_dir: Path) -> dict:
    if sample_count <= 0:
        return {
            "dataset": "FUNSD",
            "metric": "key_value_match_rate",
            "sample_count": 0,
            "score": 0.0,
            "target": 0.85,
            "cases": [],
        }

    from datasets import load_dataset

    from agent_file_create.document.extractor import extract_from_file

    ds = load_dataset("jinho8345/funsd", split=f"test[:{sample_count}]")
    cases = []
    total_pairs = 0
    matched_pairs = 0

    for idx, item in enumerate(ds):
        print(f"[FUNSD] {idx + 1}/{len(ds)}")
        img_path = work_dir / "funsd" / f"funsd_{idx}.png"
        _save_image(item["img"], img_path)
        result = extract_from_file(str(img_path), preprocess=True)
        kv_pairs = _funsd_key_value_pairs(item)
        joined = "\n".join(
            [
                str(result.get("title") or ""),
                str(result.get("summary") or ""),
                " ".join(str(x) for x in (result.get("key_points") or [])),
                str(result.get("_ocr_text") or ""),
                json.dumps(result.get("data", ""), ensure_ascii=False),
            ]
        )
        local_match = 0
        for key, value in kv_pairs[:10]:
            total_pairs += 1
            ok = _contains_answer(joined, [key]) and _contains_answer(joined, [value])
            matched_pairs += int(ok)
            local_match += int(ok)
        cases.append(
            {
                "filename": item.get("filename"),
                "pair_count_eval": min(len(kv_pairs), 10),
                "matched_pairs": local_match,
                "pred_summary": str(result.get("summary") or "")[:180],
            }
        )

    return {
        "dataset": "FUNSD",
        "metric": "key_value_match_rate",
        "sample_count": len(cases),
        "score": round(matched_pairs / float(total_pairs or 1), 4),
        "target": 0.85,
        "cases": cases,
    }


def _fetch_tabfact_rows(length: int) -> list[dict]:
    import requests

    base = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/master"
    val_examples = requests.get(f"{base}/tokenized_data/val_examples.json", timeout=120)
    val_examples.raise_for_status()
    payload = val_examples.json()

    rows: list[dict] = []
    for table_id, item in list(payload.items())[:length]:
        statements = item[0] if len(item) > 0 else []
        labels = item[1] if len(item) > 1 else []
        caption = item[2] if len(item) > 2 else ""
        csv_resp = requests.get(f"{base}/data/all_csv/{table_id}", timeout=120)
        csv_resp.raise_for_status()
        rows.append(
            {
                "table_id": table_id,
                "table_text": csv_resp.text,
                "table_caption": caption,
                "statement": statements[0] if statements else "",
                "label": labels[0] if labels else 0,
            }
        )
    return rows


def evaluate_tabfact(root: Path, sample_count: int, work_dir: Path) -> dict:
    if sample_count <= 0:
        return {
            "dataset": "TabFact",
            "metric": "table_header_preservation",
            "sample_count": 0,
            "score": 0.0,
            "target": 0.90,
            "cases": [],
        }

    from agent_file_create.document.extractor import extract_from_file

    rows = _fetch_tabfact_rows(sample_count)
    cases = []
    row_ok = 0
    total = 0
    for idx, row in enumerate(rows):
        print(f"[TabFact] {idx + 1}/{len(rows)}")
        table_md = _tabfact_table_markdown(row.get("table_text", ""))
        content = "\n".join(
            [
                f"标题: {row.get('table_caption', '')}",
                "表格内容:",
                table_md,
                f"陈述: {row.get('statement', '')}",
                f"标签: {'entailed' if int(row.get('label', 0)) == 1 else 'refuted'}",
            ]
        )
        txt_path = work_dir / "tabfact" / f"tabfact_{idx}.txt"
        _write_text_file(txt_path, content)
        result = extract_from_file(str(txt_path), preprocess=True)
        summary_text = "\n".join(
            [
                str(result.get("title") or ""),
                str(result.get("summary") or ""),
                " ".join(str(x) for x in (result.get("key_points") or [])),
                json.dumps(result.get("data", ""), ensure_ascii=False),
            ]
        )
        header = table_md.splitlines()[0] if table_md else ""
        header_cells = [cell.strip() for cell in header.split("|") if cell.strip()]
        matched_headers = sum(1 for cell in header_cells if _contains_answer(summary_text, [cell]))
        ok = bool(header_cells and matched_headers >= max(1, int(len(header_cells) * 0.6)))
        total += 1
        row_ok += int(ok)
        cases.append(
            {
                "table_id": row.get("table_id"),
                "statement": row.get("statement", ""),
                "label": row.get("label"),
                "pred_summary": str(result.get("summary") or "")[:180],
                "matched_header_cells": matched_headers,
                "header_cell_count": len(header_cells),
                "header_preserved": ok,
            }
        )
    return {
        "dataset": "TabFact",
        "metric": "table_header_preservation",
        "sample_count": len(cases),
        "score": round(row_ok / float(total or 1), 4),
        "target": 0.90,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q1 benchmark evaluation with local models")
    parser.add_argument("--docvqa", type=int, default=10)
    parser.add_argument("--funsd", type=int, default=10)
    parser.add_argument("--tabfact", type=int, default=10)
    parser.add_argument("--text-model", default="qwen3.5:9b")
    parser.add_argument("--vision-model", default="minicpm-v:8b")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--output", default="result/q1_benchmark_eval.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    work_dir = root / "result" / "q1_benchmark_samples"
    work_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MODEL_NAME"] = args.text_model
    os.environ["EXTRACT_MODEL_NAME"] = args.text_model
    os.environ["VISION_MODEL_NAME"] = args.vision_model
    os.environ["OLLAMA_HOST"] = args.endpoint
    os.environ["EXTRACT_API_STYLE"] = "ollama"
    os.environ["OCR_ENABLED"] = "true"

    sys.path.insert(0, str(root))

    started = time.time()
    reports = [
        evaluate_docvqa(root, args.docvqa, work_dir),
        evaluate_funsd(root, args.funsd, work_dir),
        evaluate_tabfact(root, args.tabfact, work_dir),
    ]
    summary = {
        "text_model": args.text_model,
        "vision_model": args.vision_model,
        "seconds": round(time.time() - started, 2),
        "reports": [
            {
                "dataset": r["dataset"],
                "metric": r["metric"],
                "sample_count": r["sample_count"],
                "score": r["score"],
                "target": r["target"],
                "pass": bool(r["score"] >= r["target"]),
            }
            for r in reports
        ],
    }

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "details": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
