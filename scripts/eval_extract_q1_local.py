from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _default_samples(root: Path) -> list[str]:
    candidates = [
        root / "test_doc" / "pdf" / "e1.pdf",
        root / "test_doc" / "docx" / "1.docx",
        root / "test_doc" / "ppt" / "屏幕实验.ppt",
        root / "test_doc" / "xlsx" / "多模态表格提取测试集_完整版.xlsx",
        root / "test_doc" / "jpg" / "1.jpg",
    ]
    return [str(p) for p in candidates if p.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Q1 本地多模态提取评测脚本")
    parser.add_argument("--text-model", default="qwen3.5:9b", help="本地文本抽取模型")
    parser.add_argument("--vision-model", default="minicpm-v:8b", help="本地视觉模型")
    parser.add_argument("--endpoint", default="http://localhost:11434", help="Ollama endpoint")
    parser.add_argument("--samples", nargs="*", default=None, help="待测文件列表")
    parser.add_argument(
        "--output",
        default="result/q1_extract_eval_local.json",
        help="评测结果输出路径",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.environ["MODEL_NAME"] = args.text_model
    os.environ["EXTRACT_MODEL_NAME"] = args.text_model
    os.environ["VISION_MODEL_NAME"] = args.vision_model
    os.environ["OLLAMA_HOST"] = args.endpoint
    os.environ["EXTRACT_API_STYLE"] = "ollama"
    os.environ["OCR_ENABLED"] = "true"

    sys.path.insert(0, str(root))

    from agent_file_create.document.extractor import (
        ab_extract,
        deduplicate_extracted_results,
        extract_from_file,
    )
    from agent_file_create.preprocessor import compute_quality_metrics

    sample_files = args.samples or _default_samples(root)
    if not sample_files:
        print("未找到默认样本，请通过 --samples 指定文件。")
        return 1

    report_items: list[dict] = []
    plain_results: list[dict] = []

    for file_path in sample_files:
        p = Path(file_path)
        t0 = time.perf_counter()
        ext = p.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".pdf"}:
            outcome = ab_extract(str(p))
            chosen = outcome["chosen"]
            metrics = compute_quality_metrics(chosen)
            elapsed = time.perf_counter() - t0
            item = {
                "file": p.name,
                "path": str(p),
                "mode": "ab_extract",
                "elapsed_s": round(elapsed, 2),
                "content_type": chosen.get("content_type"),
                "method": chosen.get("_extraction_method", ""),
                "required_ok": metrics.get("required_ok"),
                "filled_fields": metrics.get("filled_fields"),
                "field_ratio": metrics.get("field_ratio"),
                "missing_required": metrics.get("missing_required"),
                "has_tables": bool(chosen.get("_has_tables")),
                "has_ocr": bool(chosen.get("_has_ocr")),
                "title": chosen.get("title", ""),
                "summary_preview": str(chosen.get("summary") or "")[:160],
                "ab_metrics": {
                    "a": outcome.get("a"),
                    "b": outcome.get("b"),
                },
            }
            plain_results.append(chosen)
        else:
            chosen = extract_from_file(str(p), preprocess=True)
            metrics = compute_quality_metrics(chosen)
            elapsed = time.perf_counter() - t0
            item = {
                "file": p.name,
                "path": str(p),
                "mode": "extract",
                "elapsed_s": round(elapsed, 2),
                "content_type": chosen.get("content_type"),
                "method": chosen.get("_extraction_method", ""),
                "required_ok": metrics.get("required_ok"),
                "filled_fields": metrics.get("filled_fields"),
                "field_ratio": metrics.get("field_ratio"),
                "missing_required": metrics.get("missing_required"),
                "has_tables": bool(chosen.get("_has_tables")),
                "has_ocr": bool(chosen.get("_has_ocr")),
                "title": chosen.get("title", ""),
                "summary_preview": str(chosen.get("summary") or "")[:160],
            }
            plain_results.append(chosen)

        if chosen.get("error"):
            item["error"] = str(chosen.get("error"))
        report_items.append(item)
        print(
            f"- {p.name}: required_ok={item['required_ok']} "
            f"filled={item['filled_fields']}/7 elapsed={item['elapsed_s']}s"
        )

    deduped = deduplicate_extracted_results(plain_results)
    ok_items = [x for x in report_items if not x.get("error")]
    summary = {
        "text_model": args.text_model,
        "vision_model": args.vision_model,
        "endpoint": args.endpoint,
        "sample_count": len(report_items),
        "success_count": len(ok_items),
        "error_count": len(report_items) - len(ok_items),
        "required_ok_count": sum(1 for x in ok_items if x.get("required_ok")),
        "avg_filled_fields": round(
            sum(float(x.get("filled_fields") or 0) for x in ok_items) / float(len(ok_items) or 1),
            2,
        ),
        "deduped_count": len(deduped),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "items": report_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n评测汇总：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
