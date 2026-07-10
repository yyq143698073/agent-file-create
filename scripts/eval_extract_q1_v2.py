from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import statistics
import sys
import tarfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _norm_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_pair_text(text: str) -> str:
    s = _norm_text(text)
    # Collapse common separators / punctuation.
    s = re.sub(r"[_/\\|:\-.,;]+", " ", s)
    # Keep only word chars + CJK.
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return round(values[0], 4), 0.0
    return round(statistics.mean(values), 4), round(statistics.pstdev(values), 4)


def _mean_seconds(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.mean(values), 2)


def _save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(path)
        return
    if isinstance(image, dict) and image.get("bytes") is not None:
        path.write_bytes(image["bytes"])
        return
    raise TypeError("unsupported image type")


def _write_image_bytes(raw: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _extract_from_file(file_path: str) -> dict[str, Any]:
    from agent_file_create.document.extractor import extract_from_file

    return extract_from_file(file_path, preprocess=True)


def _prepare_image_bytes(file_path: str) -> bytes:
    from agent_file_create.config import IMAGE_JPEG_QUALITY, IMAGE_MAX_LONG_EDGE
    from agent_file_create.preprocessor import preprocess_image_path

    return preprocess_image_path(
        file_path,
        max_long_edge=IMAGE_MAX_LONG_EDGE,
        jpeg_quality=IMAGE_JPEG_QUALITY,
        profile="adaptive",
    )


def _ocr_text_for_image_bytes(image_bytes: bytes) -> str:
    from agent_file_create.preprocessor import easyocr_image, merge_ocr_texts, ocr_image

    primary = ocr_image(image_bytes)
    backup = easyocr_image(image_bytes)
    return merge_ocr_texts(primary, backup, max_lines=120)


def _extract_docvqa_answer(file_path: str, question: str) -> str:
    from agent_file_create.config import EXTRACT_API_STYLE, VISION_MODEL_NAME
    from agent_file_create.llm_client import call_llm

    image_bytes = _prepare_image_bytes(file_path)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    ocr_text = _ocr_text_for_image_bytes(image_bytes)
    ocr_hint = ""
    if ocr_text:
        ocr_hint = (
            "\nOCR 预识别文本如下，优先从中定位可核对实体、编号、日期、金额、地址，再结合视觉确认：\n"
            + ocr_text[:3500]
        )
    prompt = (
        "你在执行 DocVQA 文档视觉问答。"
        "请只回答问题本身，输出尽量短的答案短语，不要输出 JSON，不要解释，不要复述题目。"
        "如果答案是文档中的字段值，请尽量逐字保留原文。"
        "若无法确定，输出空字符串。"
        f"\n问题：{question.strip()}{ocr_hint}"
    )
    raw = call_llm(
        prompt,
        images_base64=[image_b64],
        timeout_s=90,
        api_style=EXTRACT_API_STYLE,
        model_name=VISION_MODEL_NAME,
        system="你是严格的文档视觉问答助手。",
    )
    answer = str(raw or "").strip()
    answer = re.sub(r"^```(?:text)?\s*", "", answer, flags=re.I).strip()
    answer = re.sub(r"\s*```$", "", answer).strip()
    answer = re.sub(r"^(答案|answer)[:：]\s*", "", answer, flags=re.I).strip()
    return answer


def _extract_ocr_text(file_path: str) -> str:
    image_bytes = _prepare_image_bytes(file_path)
    return _ocr_text_for_image_bytes(image_bytes)


def _set_runtime_env(text_model: str, vision_model: str, endpoint: str, ablate: set[str]) -> None:
    os.environ["MODEL_NAME"] = text_model
    os.environ["EXTRACT_MODEL_NAME"] = text_model
    os.environ["VISION_MODEL_NAME"] = vision_model
    os.environ["OLLAMA_HOST"] = endpoint
    os.environ["EXTRACT_API_STYLE"] = "ollama"
    os.environ["OCR_ENABLED"] = "true"

    toggles = {
        "Q1_ENABLE_SKEW": "skew" not in ablate,
        "Q1_ENABLE_BINARIZE": "binarize" not in ablate,
        "Q1_ENABLE_DENOISE": "denoise" not in ablate,
        "Q1_ENABLE_BOUNDARY": "boundary" not in ablate,
        "Q1_ENABLE_LLM_OCR_FIX": "llm_fix" not in ablate,
        "Q1_ENABLE_TABLE_STRUCT": "table_struct" not in ablate,
    }
    for key, enabled in toggles.items():
        os.environ[key] = "true" if enabled else "false"


def _extract_pred_pairs(result: dict[str, Any]) -> list[tuple[str, str]]:
    sources: list[Any] = []
    data = result.get("data")
    if data is not None:
        sources.append(data)
    for key in ("raw_fields", "normalized_fields", "typed_fields", "reading_order_fields"):
        if key in result:
            sources.append(result.get(key))
        data_obj = result.get("data")
        if isinstance(data_obj, dict) and key in data_obj:
            sources.append(data_obj.get(key))

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            field = node.get("field")
            value = node.get("value")
            if field is not None and value is not None:
                token = (_norm_pair_text(str(field)), _norm_pair_text(str(value)))
                if token[0] and token[1] and token not in seen:
                    seen.add(token)
                    pairs.append(token)
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    for source in sources:
        _visit(source)
    return pairs


def _extract_answer_text(result: dict[str, Any]) -> str:
    parts = [
        str(result.get("title") or ""),
        str(result.get("summary") or ""),
        " ".join(str(x) for x in (result.get("key_points") or [])),
        str(result.get("_ocr_text") or ""),
    ]
    data = result.get("data")
    if data is not None:
        parts.append(json.dumps(data, ensure_ascii=False))
    return "\n".join(part for part in parts if part)


def _best_anls(prediction: str, answers: list[str]) -> float:
    pred = _norm_text(prediction)
    if not pred:
        return 0.0
    best = 0.0
    for answer in answers:
        gold = _norm_text(answer)
        if not gold:
            continue
        dist = _levenshtein(pred, gold)
        score = 1.0 - (dist / float(max(len(pred), len(gold), 1)))
        best = max(best, score)
    return max(0.0, round(best, 4))


def _html_to_tokens(html_text: str, ignore_nodes: set[str] | None = None) -> list[str]:
    from lxml import etree, html

    ignore_nodes = ignore_nodes or set()
    wrapped = f"<root>{html_text}</root>"
    root = html.fromstring(wrapped)
    tokens: list[str] = []

    def _walk(node: Any) -> None:
        tag = str(node.tag).lower() if hasattr(node, "tag") else ""
        if tag in ignore_nodes:
            for child in node:
                _walk(child)
            return
        if tag and tag != "root":
            attr_items = sorted((k, v) for k, v in node.attrib.items() if k in {"rowspan", "colspan"})
            attr_text = " ".join(f"{k}={v}" for k, v in attr_items)
            tokens.append(f"<{tag}{(' ' + attr_text) if attr_text else ''}>")
        text = _norm_text(node.text or "")
        if text:
            tokens.extend(text.split())
        for child in node:
            _walk(child)
            tail = _norm_text(child.tail or "")
            if tail:
                tokens.extend(tail.split())
        if tag and tag != "root":
            tokens.append(f"</{tag}>")

    _walk(root)
    return tokens


def _teds_score(pred_html: str, gold_html: str) -> float:
    pred_tokens = _html_to_tokens(pred_html, ignore_nodes={"b"})
    gold_tokens = _html_to_tokens(gold_html, ignore_nodes={"b"})
    if not pred_tokens and not gold_tokens:
        return 1.0
    dist = _levenshtein("\n".join(pred_tokens), "\n".join(gold_tokens))
    base = max(len("\n".join(pred_tokens)), len("\n".join(gold_tokens)), 1)
    return max(0.0, round(1.0 - (dist / float(base)), 4))


def _pairs_to_prf(pred_pairs: list[tuple[str, str]], gold_pairs: list[tuple[str, str]]) -> tuple[float, float, float, int]:
    pred_set = set(pred_pairs)
    gold_set = set(gold_pairs)
    matched = len(pred_set & gold_set)
    precision = matched / float(len(pred_set) or 1)
    recall = matched / float(len(gold_set) or 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return round(precision, 4), round(recall, 4), round(f1, 4), matched


def _cer_score(pred_text: str, gold_text: str) -> float:
    pred = _norm_text(pred_text).replace(" ", "")
    gold = _norm_text(gold_text).replace(" ", "")
    if not gold:
        return 0.0
    return round(_levenshtein(pred, gold) / float(len(gold) or 1), 4)


def _line_accuracy(pred_lines: list[str], gold_lines: list[str]) -> float:
    if not gold_lines:
        return 0.0
    pred_norm = [_norm_text(x) for x in pred_lines if _norm_text(x)]
    gold_norm = [_norm_text(x) for x in gold_lines if _norm_text(x)]
    hits = sum(1 for line in gold_norm if line in pred_norm)
    return round(hits / float(len(gold_norm) or 1), 4)


def _load_docvqa(sample_count: int):
    from datasets import load_dataset

    prepared_dir = ROOT / "result" / "q1_eval_v2" / "samples" / "docvqa"
    gt_path = prepared_dir / "ground_truth.json"
    image_dir = prepared_dir / "images"
    if gt_path.exists() and image_dir.exists():
        rows = json.loads(gt_path.read_text(encoding="utf-8"))
        for item in rows:
            item["image_path"] = str(image_dir / str(item.get("file") or ""))
        return rows[:sample_count]
    return load_dataset("nielsr/docvqa_1200_examples", split=f"test[:{sample_count}]")


def _load_funsd(sample_count: int):
    from datasets import load_dataset

    prepared_dir = ROOT / "result" / "q1_eval_v2" / "samples" / "funsd"
    gt_path = prepared_dir / "ground_truth.json"
    image_dir = prepared_dir / "images"
    if gt_path.exists() and image_dir.exists():
        rows = json.loads(gt_path.read_text(encoding="utf-8"))
        for item in rows:
            item["image_path"] = str(image_dir / str(item.get("file") or ""))
        return rows[:sample_count]
    return load_dataset("jinho8345/funsd", split=f"test[:{sample_count}]")


def _funsd_gold_pairs(item: dict[str, Any]) -> list[tuple[str, str]]:
    if isinstance(item.get("pairs"), list):
        pairs: list[tuple[str, str]] = []
        for pair in item.get("pairs") or []:
            field = _norm_pair_text(str((pair or {}).get("field") or ""))
            value = _norm_pair_text(str((pair or {}).get("value") or ""))
            if field and value:
                pairs.append((field, value))
        return pairs
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
            pairs.append((_norm_pair_text(current_q), _norm_pair_text(text)))
            current_q = ""
    return pairs


def _sroie_gold_pairs(item: dict[str, Any]) -> list[tuple[str, str]]:
    gold_pairs: list[tuple[str, str]] = []
    if "ground_truth" in item and isinstance(item["ground_truth"], dict):
        source = item["ground_truth"]
    elif "entities" in item and isinstance(item["entities"], dict):
        source = item["entities"]
    else:
        source = item
    for key in ("company", "date", "address", "total"):
        value = source.get(key)
        if value:
            gold_pairs.append((_norm_pair_text(key), _norm_pair_text(str(value))))
    return gold_pairs


def _ctw_gold_text(item: dict[str, Any]) -> tuple[str, list[str]]:
    lines: list[str] = []
    annotations = item.get("annotations")
    if isinstance(annotations, list):
        for sentence in annotations:
            if isinstance(sentence, dict):
                text = sentence.get("text") or sentence.get("transcription") or ""
                if text:
                    lines.append(str(text))
    elif isinstance(item.get("text"), list):
        lines = [str(x) for x in item.get("text", []) if str(x).strip()]
    joined = "\n".join(lines)
    return joined, lines


def _find_local_dataset_root(default_name: str, explicit_root: str | None) -> Path | None:
    if explicit_root:
        path = Path(explicit_root)
        return path if path.exists() else None
    samples_root = ROOT / "result" / "q1_eval_v2" / "samples" / default_name
    if samples_root.exists():
        return samples_root
    datasets_root = ROOT / "result" / "q1_eval_v2" / "datasets"
    candidate = datasets_root / default_name
    if candidate.exists():
        return candidate
    if datasets_root.exists():
        for path in datasets_root.iterdir():
            if path.name.lower() == default_name.lower():
                return path
            if default_name == "sroie" and "sroie" in path.name.lower():
                return path
            if default_name == "pubtabnet" and "pubtab" in path.name.lower():
                return path
            if default_name == "ctw" and "ctw" in path.name.lower():
                return path
    return None


def _load_sroie_parquet(root: Path, sample_count: int) -> list[dict[str, Any]]:
    import pandas as pd

    candidates = list(root.glob("**/test-*.parquet")) or list(root.glob("**/train-*.parquet"))
    if not candidates:
        raise FileNotFoundError("SROIE parquet not found")
    df = pd.read_parquet(candidates[0])
    rows = df.head(sample_count).to_dict("records")
    return rows


def _parse_sroie_entity_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        k = key.strip().lower()
        if k in {"company", "date", "address", "total"}:
            out[k] = value.strip()
    return out


def _load_sroie_files(root: Path, sample_count: int) -> list[dict[str, Any]]:
    image_files = sorted(
        list(root.glob("**/*.jpg")) + list(root.glob("**/*.jpeg")) + list(root.glob("**/*.png"))
    )
    rows: list[dict[str, Any]] = []
    for image_path in image_files:
        stem = image_path.stem
        parent = image_path.parent
        txt_candidates = [
            image_path.with_suffix(".txt"),
            parent / f"{stem}.txt",
            parent.parent / "entities" / f"{stem}.txt",
            parent.parent / "entity" / f"{stem}.txt",
            parent.parent / "annotations" / f"{stem}.txt",
        ]
        entity_data: dict[str, Any] = {}
        for txt_path in txt_candidates:
            if txt_path.exists():
                entity_data = _parse_sroie_entity_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
                if entity_data:
                    break
        if not entity_data:
            continue
        rows.append({"image_path": str(image_path), "entities": entity_data})
        if len(rows) >= sample_count:
            break
    if not rows:
        raise FileNotFoundError("SROIE images/entities not found")
    return rows


def _load_sroie_rows(root: Path, sample_count: int) -> list[dict[str, Any]]:
    gt_path = root / "ground_truth.json"
    image_dir = root / "images"
    if gt_path.exists() and image_dir.exists():
        rows = json.loads(gt_path.read_text(encoding="utf-8"))
        for item in rows:
            item["image_path"] = str(image_dir / str(item.get("file") or ""))
        return rows[:sample_count]
    try:
        return _load_sroie_parquet(root, sample_count)
    except Exception:
        return _load_sroie_files(root, sample_count)


def _load_pubtabnet_ocrflux(root: Path, sample_count: int) -> tuple[list[dict[str, Any]], Path | None]:
    gt_path = root / "ground_truth.json"
    images_dir = root / "images"
    if gt_path.exists() and images_dir.exists():
        rows = json.loads(gt_path.read_text(encoding="utf-8"))
        prepared_rows = []
        for item in rows[:sample_count]:
            prepared_rows.append(
                {
                    "image_name": str(item.get("file") or ""),
                    "gt_table": str(item.get("html_table") or ""),
                    "type": str(item.get("split") or ""),
                }
            )
        return prepared_rows, images_dir
    jsonl_path = root / "data.jsonl"
    images_dir = root / "images"
    tar_path = root / "images.tar.gz"
    if not jsonl_path.exists():
        raise FileNotFoundError("PubTabNet data.jsonl not found")
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= sample_count:
                break
            row = json.loads(line)
            rows.append(row)
    if images_dir.exists():
        return rows, images_dir
    if tar_path.exists():
        return rows, tar_path
    raise FileNotFoundError("PubTabNet images directory or tarball not found")


def _load_ctw_jsonl(root: Path, sample_count: int) -> tuple[list[dict[str, Any]], Path]:
    image_dir = root / "train_val"
    label_path = root / "val.jsonl"
    if not label_path.exists():
        label_path = root / "train.jsonl"
    if not image_dir.exists() or not label_path.exists():
        raise FileNotFoundError("CTW train_val/ or *.jsonl missing")
    rows = []
    with label_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= sample_count:
                break
            rows.append(json.loads(line))
    return rows, image_dir


def evaluate_docvqa(sample_count: int, sample_root: Path) -> dict[str, Any]:
    ds = _load_docvqa(sample_count)
    scores: list[float] = []
    seconds_list: list[float] = []
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(ds):
        print(f"[DocVQA] {idx + 1}/{len(ds)}")
        if item.get("image_path"):
            img_path = Path(str(item["image_path"]))
        else:
            img_path = sample_root / "docvqa" / f"docvqa_{idx}.png"
            _save_image(item["image"], img_path)
        start = time.time()
        question = str(item.get("question") or ((item.get("query") or {}).get("en") or "")).strip()
        prediction = _extract_docvqa_answer(str(img_path), question)
        elapsed = round(time.time() - start, 2)
        seconds_list.append(elapsed)
        answers = [str(a) for a in item.get("answers", []) if str(a).strip()]
        score = _best_anls(prediction, answers)
        scores.append(score)
        cases.append(
            {
                "id": item.get("id"),
                "question": question,
                "answers": answers[:3],
                "prediction": prediction[:300],
                "anls": score,
                "seconds": elapsed,
            }
        )
    mean, std = _mean_std(scores)
    return {
        "dataset": "docvqa",
        "metric": "anls",
        "sample_count": len(cases),
        "anls": mean,
        "anls_std": std,
        "avg_seconds": _mean_seconds(seconds_list),
        "target": 0.65,
        "pass": mean >= 0.65,
        "details": cases,
    }


def evaluate_funsd(sample_count: int, sample_root: Path) -> dict[str, Any]:
    ds = _load_funsd(sample_count)
    f1_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    seconds_list: list[float] = []
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(ds):
        print(f"[FUNSD] {idx + 1}/{len(ds)}")
        if item.get("image_path"):
            img_path = Path(str(item["image_path"]))
        else:
            img_path = sample_root / "funsd" / f"funsd_{idx}.png"
            _save_image(item["img"], img_path)
        start = time.time()
        result = _extract_from_file(str(img_path))
        elapsed = round(time.time() - start, 2)
        seconds_list.append(elapsed)
        pred_pairs = _extract_pred_pairs(result)
        gold_pairs = _funsd_gold_pairs(item)
        precision, recall, f1, matched = _pairs_to_prf(pred_pairs, gold_pairs)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        cases.append(
            {
                "filename": item.get("filename") or item.get("file"),
                "gold_pairs": len(gold_pairs),
                "pred_pairs": len(pred_pairs),
                "matched_pairs": matched,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "seconds": elapsed,
            }
        )
    p_mean, p_std = _mean_std(precision_scores)
    r_mean, r_std = _mean_std(recall_scores)
    f_mean, f_std = _mean_std(f1_scores)
    return {
        "dataset": "funsd",
        "metric": "kv_f1",
        "sample_count": len(cases),
        "kv_precision": p_mean,
        "kv_precision_std": p_std,
        "kv_recall": r_mean,
        "kv_recall_std": r_std,
        "kv_f1": f_mean,
        "kv_f1_std": f_std,
        "avg_seconds": _mean_seconds(seconds_list),
        "target": 0.60,
        "pass": f_mean >= 0.60,
        "details": cases,
    }


def evaluate_sroie(sample_count: int, sample_root: Path, dataset_root: Path) -> dict[str, Any]:
    rows = _load_sroie_rows(dataset_root, sample_count)
    f1_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    seconds_list: list[float] = []
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(rows):
        print(f"[SROIE] {idx + 1}/{len(rows)}")
        img_path = sample_root / "sroie" / f"sroie_{idx}.jpg"
        if item.get("image") is not None:
            _save_image(item["image"], img_path)
        elif item.get("image_path"):
            src = Path(str(item["image_path"]))
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(src.read_bytes())
        else:
            raise ValueError("SROIE row missing image source")
        start = time.time()
        result = _extract_from_file(str(img_path))
        elapsed = round(time.time() - start, 2)
        seconds_list.append(elapsed)
        pred_pairs = _extract_pred_pairs(result)
        gold_pairs = _sroie_gold_pairs(item)
        precision, recall, f1, matched = _pairs_to_prf(pred_pairs, gold_pairs)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        cases.append(
            {
                "index": idx,
                "gold_pairs": len(gold_pairs),
                "pred_pairs": len(pred_pairs),
                "matched_pairs": matched,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "seconds": elapsed,
            }
        )
    p_mean, p_std = _mean_std(precision_scores)
    r_mean, r_std = _mean_std(recall_scores)
    f_mean, f_std = _mean_std(f1_scores)
    return {
        "dataset": "sroie",
        "metric": "kv_f1",
        "sample_count": len(cases),
        "kv_precision": p_mean,
        "kv_precision_std": p_std,
        "kv_recall": r_mean,
        "kv_recall_std": r_std,
        "kv_f1": f_mean,
        "kv_f1_std": f_std,
        "avg_seconds": _mean_seconds(seconds_list),
        "target": 0.70,
        "pass": f_mean >= 0.70,
        "details": cases,
    }


def evaluate_pubtabnet(sample_count: int, sample_root: Path, dataset_root: Path) -> dict[str, Any]:
    rows, image_source = _load_pubtabnet_ocrflux(dataset_root, sample_count)
    tar_obj = tarfile.open(image_source, "r:gz") if image_source and image_source.suffixes[-2:] == [".tar", ".gz"] else None
    scores: list[float] = []
    seconds_list: list[float] = []
    cases: list[dict[str, Any]] = []
    try:
        for idx, item in enumerate(rows):
            print(f"[PubTabNet] {idx + 1}/{len(rows)}")
            image_name = str(item.get("image_name") or "")
            if not image_name:
                continue
            img_path = sample_root / "pubtabnet" / image_name
            if image_source and image_source.is_dir():
                src = image_source / image_name
                if not src.exists():
                    raise FileNotFoundError(f"PubTabNet image missing: {src}")
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.write_bytes(src.read_bytes())
            elif tar_obj is not None:
                member = tar_obj.getmember(f"images/{image_name}") if f"images/{image_name}" in tar_obj.getnames() else tar_obj.getmember(image_name)
                raw = tar_obj.extractfile(member)
                if raw is None:
                    raise FileNotFoundError(f"PubTabNet image missing in tar: {image_name}")
                _write_image_bytes(raw.read(), img_path)
            else:
                raise FileNotFoundError("PubTabNet image source unavailable")

            start = time.time()
            result = _extract_from_file(str(img_path))
            elapsed = round(time.time() - start, 2)
            seconds_list.append(elapsed)
            tables = result.get("_structured_tables") or []
            pred_html = ""
            if isinstance(tables, list) and tables:
                pred_html = str((tables[0] or {}).get("markdown") or "")
            if not pred_html:
                pred_html = str(result.get("_table_preview") or result.get("summary") or "")
            score = _teds_score(pred_html, str(item.get("gt_table") or ""))
            scores.append(score)
            cases.append(
                {
                    "image_name": image_name,
                    "type": item.get("type"),
                    "teds": score,
                    "seconds": elapsed,
                }
            )
    finally:
        if tar_obj is not None:
            tar_obj.close()
    mean, std = _mean_std(scores)
    return {
        "dataset": "pubtabnet",
        "metric": "teds",
        "sample_count": len(cases),
        "teds": mean,
        "teds_std": std,
        "avg_seconds": _mean_seconds(seconds_list),
        "target": 0.75,
        "pass": mean >= 0.75,
        "details": cases,
    }


def evaluate_ctw(sample_count: int, sample_root: Path, dataset_root: Path) -> dict[str, Any]:
    rows, image_dir = _load_ctw_jsonl(dataset_root, sample_count)
    cer_scores: list[float] = []
    line_scores: list[float] = []
    seconds_list: list[float] = []
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(rows):
        print(f"[CTW] {idx + 1}/{len(rows)}")
        file_name = str(item.get("file_name") or item.get("image_id") or "")
        if not file_name:
            continue
        img_path = image_dir / file_name
        if not img_path.exists():
            continue
        start = time.time()
        pred_text = _extract_ocr_text(str(img_path))
        elapsed = round(time.time() - start, 2)
        seconds_list.append(elapsed)
        gold_text, gold_lines = _ctw_gold_text(item)
        pred_lines = [line for line in pred_text.splitlines() if line.strip()]
        cer = _cer_score(pred_text, gold_text)
        line_acc = _line_accuracy(pred_lines, gold_lines)
        cer_scores.append(cer)
        line_scores.append(line_acc)
        cases.append(
            {
                "file_name": file_name,
                "cer": cer,
                "line_accuracy": line_acc,
                "seconds": elapsed,
            }
        )
    cer_mean, cer_std = _mean_std(cer_scores)
    line_mean, line_std = _mean_std(line_scores)
    return {
        "dataset": "ctw",
        "metric": "cer",
        "sample_count": len(cases),
        "cer": cer_mean,
        "cer_std": cer_std,
        "line_accuracy": line_mean,
        "line_accuracy_std": line_std,
        "avg_seconds": _mean_seconds(seconds_list),
        "target": 0.15,
        "pass": cer_mean <= 0.15 if cases else None,
        "details": cases,
    }


def _dataset_error(name: str, metric: str, exc: Exception) -> dict[str, Any]:
    return {
        "dataset": name,
        "metric": metric,
        "sample_count": 0,
        "error": str(exc),
        "details": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q1 v2 evaluation with structured metrics")
    parser.add_argument("--datasets", default="pubtabnet,funsd,sroie,docvqa,ctw")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--text-model", default="qwen3.5:9b")
    parser.add_argument("--vision-model", default="minicpm-v:8b")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--ablate", default="")
    parser.add_argument("--model-ceiling", default="")
    parser.add_argument("--output", default="result/q1_eval_v2/q1_eval_v2_baseline.json")
    parser.add_argument("--pubtabnet-root", default="")
    parser.add_argument("--sroie-root", default="")
    parser.add_argument("--ctw-root", default="")
    args = parser.parse_args()

    requested = [x.strip().lower() for x in str(args.datasets or "").split(",") if x.strip()]
    ablate = {x.strip().lower() for x in str(args.ablate or "").split(",") if x.strip()}
    sample_root = ROOT / "result" / "q1_eval_v2" / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    _set_runtime_env(args.text_model, args.vision_model, args.endpoint, ablate)

    started = time.time()
    details: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    eval_plan: list[tuple[str, Any]] = []
    if "pubtabnet" in requested:
        pub_root = _find_local_dataset_root("pubtabnet", args.pubtabnet_root)
        eval_plan.append(("pubtabnet", lambda: evaluate_pubtabnet(min(args.sample, 200), sample_root, pub_root) if pub_root else (_ for _ in ()).throw(FileNotFoundError("PubTabNet local dataset missing"))))
    if "funsd" in requested:
        eval_plan.append(("funsd", lambda: evaluate_funsd(max(50, min(args.sample, 100)), sample_root)))
    if "sroie" in requested:
        sroie_root = _find_local_dataset_root("sroie", args.sroie_root)
        eval_plan.append(("sroie", lambda: evaluate_sroie(max(50, min(args.sample, 100)), sample_root, sroie_root) if sroie_root else (_ for _ in ()).throw(FileNotFoundError("SROIE local dataset missing"))))
    if "docvqa" in requested:
        eval_plan.append(("docvqa", lambda: evaluate_docvqa(max(50, min(args.sample, 100)), sample_root)))
    if "ctw" in requested:
        ctw_root = _find_local_dataset_root("ctw", args.ctw_root)
        eval_plan.append(("ctw", lambda: evaluate_ctw(min(args.sample, 200), sample_root, ctw_root) if ctw_root else (_ for _ in ()).throw(FileNotFoundError("CTW local dataset missing"))))

    for name, fn in eval_plan:
        try:
            report = fn()
        except Exception as exc:
            metric = {"pubtabnet": "teds", "funsd": "kv_f1", "sroie": "kv_f1", "docvqa": "anls", "ctw": "cer"}.get(name, "score")
            report = _dataset_error(name, metric, exc)
        details[name] = report
        compact = {
            "sample_count": report.get("sample_count", 0),
            "target": report.get("target"),
            "pass": report.get("pass"),
        }
        for key in ("teds", "teds_std", "kv_precision", "kv_precision_std", "kv_recall", "kv_recall_std", "kv_f1", "kv_f1_std", "anls", "anls_std", "cer", "cer_std", "line_accuracy", "line_accuracy_std", "avg_seconds", "error"):
            if key in report:
                compact[key] = report[key]
        summary[name] = compact

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "text_model": args.text_model,
            "vision_model": args.vision_model,
            "endpoint": args.endpoint,
            "ablation": sorted(ablate),
            "model_ceiling": args.model_ceiling or "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - started, 2),
            "requested_datasets": requested,
        },
        "summary": summary,
        "details": details,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
