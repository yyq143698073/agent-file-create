#!/usr/bin/env python
"""Evaluate planner and critic with a local Ollama model on ALCE/ASQA.

Why ASQA from ALCE?
- `planner` needs a question that can be decomposed into sub-questions and then
  retrieve evidence. ASQA provides `qa_pairs` as gold sub-questions.
- `critic` needs grounded long-form answers plus supporting documents. ALCE
  provides `answer` + top-100 retrieved `docs`, making it suitable for
  evidence-based review.

This script compares:
1. Planner baseline  : single-query retrieval with fixed top-k.
2. Planner optimized : current `plan_section_knowledge()`.
3. Critic baseline   : a minimal factuality prompt.
4. Critic optimized  : current `run_critic()`.

Default model path is local Ollama:
  python scripts/eval_planner_critic_alce_local.py
  python scripts/eval_planner_critic_alce_local.py --model qwen3.5:4b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def _set_local_llm_env(model: str, endpoint: str) -> None:
    os.environ["OLLAMA_HOST"] = endpoint
    os.environ["MODEL_NAME"] = model
    os.environ["OPENAI_BASE_URL"] = endpoint
    os.environ["OPENAI_MODEL_NAME"] = model
    os.environ["OPENAI_API_ENDPOINT"] = endpoint.rstrip("/") + "/v1/chat/completions"
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ["CONTENT_API_STYLE"] = "ollama"
    os.environ["CONTENT_MODEL_NAME"] = model
    os.environ["CONTENT_API_ENDPOINT"] = endpoint
    os.environ["OUTLINE_API_STYLE"] = "ollama"
    os.environ["OUTLINE_MODEL_NAME"] = model
    os.environ["OUTLINE_API_ENDPOINT"] = endpoint
    os.environ["PLANNER_API_STYLE"] = "ollama"
    os.environ["PLANNER_MODEL_NAME"] = model
    os.environ["PLANNER_API_ENDPOINT"] = endpoint


PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in os.sys.path:
    os.sys.path.insert(0, str(PROJ))


ALCE_URL = "https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar"
ALCE_MEMBER = "ALCE-data/asqa_eval_gtr_top100.json"


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'“”‘’`]", "", text)
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9][a-z0-9\-]+", text)
    if tokens:
        return tokens
    # Fallback for non-space scripts
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    out = []
    for i in range(len(chars) - 1):
        out.append(chars[i] + chars[i + 1])
    return out[:24]


def safe_snippet(doc: dict, limit: int = 1000) -> str:
    text = str(doc.get("extraction") or doc.get("summary") or doc.get("text") or "").strip()
    return text[:limit]


def answer_appears(text: str, answers: Iterable[str]) -> bool:
    base = normalize_text(text)
    for answer in answers:
        ans = normalize_text(str(answer))
        if len(ans) >= 2 and ans in base:
            return True
    return False


def ensure_alce_tar(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        return path
    print(f"[download] ALCE-data -> {path}")
    urllib.request.urlretrieve(ALCE_URL, path)
    return path


def load_asqa_samples(alce_tar: Path) -> list[dict]:
    with tarfile.open(alce_tar) as tf:
        with tf.extractfile(ALCE_MEMBER) as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ALCE ASQA payload is not a list")
    return data


def sample_is_usable(sample: dict) -> bool:
    if not sample.get("question") or not sample.get("answer"):
        return False
    qa_pairs = sample.get("qa_pairs") or []
    docs = sample.get("docs") or []
    if len(qa_pairs) < 2 or len(docs) < 8:
        return False
    if len(str(sample.get("answer"))) < 120:
        return False
    non_empty_docs = sum(1 for d in docs[:20] if safe_snippet(d, 300))
    return non_empty_docs >= 5


@dataclass(frozen=True)
class MiniHit:
    kb: str
    doc_id: str
    chunk_id: str
    chunk_index: int
    section_path: str
    content: str
    score: float
    meta: dict
    parent_chunk_id: str = ""


class MiniKB:
    """A small lexical retriever over ALCE docs.

    It is intentionally simple: this evaluation focuses on whether planner
    prompts/query expansion improve evidence coverage over a fixed retrieved set.
    """

    def __init__(self, docs: list[dict], kb_name: str = "asqa-mini") -> None:
        self.kb_name = kb_name
        self.docs: list[MiniHit] = []
        for idx, doc in enumerate(docs):
            title = str(doc.get("title") or f"doc_{idx}")
            content = safe_snippet(doc, 1400)
            if not content:
                continue
            self.docs.append(
                MiniHit(
                    kb=kb_name,
                    doc_id=title,
                    chunk_id=f"{title}::{idx}",
                    chunk_index=idx,
                    section_path=title,
                    content=content,
                    score=0.0,
                    meta={"title": title},
                )
            )

    def _search_impl(
        self,
        query: str,
        *,
        top_k: int = 8,
        title_boost: bool = False,
    ) -> list[MiniHit]:
        q = normalize_text(query)
        q_tokens = tokenize(query)
        q_set = set(q_tokens)
        results: list[MiniHit] = []
        for doc in self.docs:
            title = normalize_text(doc.doc_id)
            text = normalize_text(doc.content)
            d_tokens = set(tokenize(doc.doc_id + " " + doc.content))
            overlap = len(q_set & d_tokens) / max(len(q_set), 1)
            exact_bonus = 0.30 if q and q in text else 0.0
            title_overlap = len(q_set & set(tokenize(doc.doc_id))) / max(len(q_set), 1)
            title_bonus = 0.25 * title_overlap if title_boost else 0.0
            title_exact = 0.20 if title_boost and q and q in title else 0.0
            score = overlap + exact_bonus + title_bonus + title_exact
            if score <= 0:
                continue
            results.append(
                MiniHit(
                    kb=doc.kb,
                    doc_id=doc.doc_id,
                    chunk_id=doc.chunk_id,
                    chunk_index=doc.chunk_index,
                    section_path=doc.section_path,
                    content=doc.content,
                    score=round(score, 4),
                    meta=doc.meta,
                )
            )
        results.sort(key=lambda h: (-h.score, h.chunk_index))
        return results[:top_k]

    def search(self, *, kb: str, query: str, top_k: int = 8) -> list[MiniHit]:
        return self._search_impl(query, top_k=top_k, title_boost=False)

    def search_adaptive(
        self,
        *,
        kb: str,
        query: str,
        top_k: int = 8,
        enable_diversity: bool = False,
        enable_title_boost: bool = False,
    ) -> list[MiniHit]:
        _ = enable_diversity
        return self._search_impl(query, top_k=top_k, title_boost=enable_title_boost)

    def search_hyde(self, *, kb: str, query: str, top_k: int = 8) -> list[MiniHit]:
        # In this lightweight harness we keep HyDE cheap and deterministic.
        return self._search_impl(query, top_k=top_k, title_boost=True)


def fast_local_rerank(query: str, sentences: list[str], top_k: int = 8) -> list[int]:
    """Cheap lexical rerank used only by this evaluation harness."""
    q_tokens = set(tokenize(query))
    scored = []
    for idx, sent in enumerate(sentences):
        s_tokens = set(tokenize(sent))
        score = len(q_tokens & s_tokens)
        if re.search(r"\d", sent):
            score += 1
        scored.append((score, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in scored[:top_k]]


def baseline_plan(sample: dict, kb: MiniKB) -> dict:
    question = str(sample["question"]).strip()
    hits = kb.search(kb=kb.kb_name, query=question, top_k=5)
    materials = " ".join(h.content for h in hits)[:1200]
    return {
        "knowledge_points": [question],
        "materials": materials,
        "_raw_hits": hits,
        "hits_count": len(hits),
    }


_SECTION_TYPE_KEYWORDS_EVAL: dict[str, list[str]] = {
    "data": [
        "experiment", "data", "result", "results", "performance", "evaluation",
        "metric", "accuracy", "recall", "f1", "bleu", "rouge", "compare",
    ],
    "experiment_setup": [
        "method", "dataset", "implementation", "model", "setting", "configuration",
        "setup", "training", "preprocess",
    ],
    "analysis": [
        "analysis", "discussion", "future", "limitation", "implication", "trend",
    ],
}


def classify_section_type_eval(section_title: str) -> str:
    title = normalize_text(section_title)
    experiment_score = sum(1 for kw in _SECTION_TYPE_KEYWORDS_EVAL["experiment_setup"] if kw in title)
    data_score = sum(1 for kw in _SECTION_TYPE_KEYWORDS_EVAL["data"] if kw in title)
    analysis_score = sum(1 for kw in _SECTION_TYPE_KEYWORDS_EVAL["analysis"] if kw in title)
    scores = [
        (experiment_score, "experiment_setup"),
        (data_score, "data"),
        (analysis_score, "analysis"),
    ]
    scores.sort(key=lambda x: (-x[0], ["experiment_setup", "data", "analysis"].index(x[1])))
    if scores[0][0] > 0:
        return scores[0][1]
    return "review"


def current_plan_fast(
    *,
    question: str,
    knowledge_points: list[str],
    kb: MiniKB,
    planner_mod,
    target_words: int = 1200,
) -> dict:
    sec_type = classify_section_type_eval(question)
    ctx_budget = planner_mod._get_context_budget(sec_type, target_words)

    all_queries: list[str] = []
    for kp in knowledge_points[:4]:
        all_queries.extend(planner_mod._generate_queries_from_knowledge_point(kp, question))
    if not all_queries:
        all_queries.extend(planner_mod._extract_concepts_from_title(question)[:3])
    all_queries.extend(planner_mod._extract_terms(question, min_len=2, max_len=6)[:2])

    seen = set()
    queries = []
    for q in all_queries:
        q = str(q).strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
        if len(queries) >= 12:
            break

    all_hits = planner_mod._multi_query_retrieve(kb, kb.kb_name, queries, sec_type, max_hits=25)
    result = {
        "knowledge_points": knowledge_points[:4],
        "section_type": sec_type,
        "hits_count": len(all_hits),
        "_raw_hits": all_hits,
        "materials": "",
    }
    if not all_hits:
        return result

    all_sentences: list[str] = []
    for h in all_hits:
        for sent in re.split(r"[。！？.!?\n]+", str(h.content or "")):
            sent = sent.strip()
            if len(sent) >= 8:
                all_sentences.append(sent)
    if not all_sentences:
        return result

    key_terms: set[str] = set()
    for kp in knowledge_points[:4]:
        key_terms.update(planner_mod._extract_terms(kp, min_len=2, max_len=6)[:5])
    key_terms.update(planner_mod._extract_terms(question, min_len=2, max_len=6)[:3])

    scored_sentences: list[tuple[int, int, str]] = []
    for idx, sent in enumerate(all_sentences):
        score = sum(1 for term in key_terms if term and term in sent)
        if re.search(r"\d|%|F1|BLEU|ROUGE|accuracy|recall", sent, re.IGNORECASE):
            score += 2
        scored_sentences.append((score, idx, sent))
    scored_sentences.sort(key=lambda x: (-x[0], x[1]))
    filtered = scored_sentences[:25]
    filtered_sentence_list = [s for _, _, s in filtered]
    if not filtered_sentence_list:
        filtered_sentence_list = all_sentences[:8]

    ranked_idx = fast_local_rerank(" ".join(knowledge_points[:2]) or question, filtered_sentence_list, top_k=8)
    materials = ""
    for ri in ranked_idx:
        if 0 <= ri < len(filtered_sentence_list):
            sent = filtered_sentence_list[ri]
            if len(materials) + len(sent) + 2 > ctx_budget:
                break
            materials += sent + ". "
    result["materials"] = materials.strip() or " ".join(filtered_sentence_list[:8])[:ctx_budget]
    return result


def build_outline(sample: dict) -> str:
    lines = ["# Answer Review"]
    for idx, pair in enumerate((sample.get("qa_pairs") or [])[:3], start=1):
        lines.append(f"## Point {idx}: {pair.get('question', '')}")
    return "\n".join(lines)


def build_materials(sample: dict, limit_docs: int = 6, limit_chars: int = 2400) -> str:
    chunks = []
    for doc in (sample.get("docs") or [])[:limit_docs]:
        title = str(doc.get("title") or "")
        text = safe_snippet(doc, 500)
        if text:
            chunks.append(f"[{title}] {text}")
    return "\n".join(chunks)[:limit_chars]


def collect_gold_answer_sets(sample: dict) -> list[list[str]]:
    out: list[list[str]] = []
    for pair in sample.get("qa_pairs") or []:
        answers = [str(a).strip() for a in (pair.get("short_answers") or []) if str(a).strip()]
        if answers:
            out.append(answers)
    return out


def critic_grounding_score(sample: dict) -> float:
    gold_sets = collect_gold_answer_sets(sample)
    materials = build_materials(sample)
    covered = sum(1 for answers in gold_sets if answer_appears(materials, answers))
    return covered / max(len(gold_sets), 1)


def critic_numeric_supported(sample: dict) -> bool:
    answer = str(sample.get("answer") or "")
    materials = build_materials(sample)
    nums = re.findall(r"\b\d[\d,\.%:-]*\b", answer)
    return all(n in materials for n in nums)


def choose_critic_samples(
    all_samples: list[dict],
    limit: int,
    seed: int,
    subset_mode: str,
) -> tuple[list[dict], int]:
    usable = [s for s in all_samples if sample_is_usable(s)]
    if subset_mode == "strict":
        eligible = [s for s in usable if critic_grounding_score(s) >= 0.999]
    elif subset_mode == "strict_numeric":
        eligible = [
            s for s in usable
            if critic_grounding_score(s) >= 0.999 and critic_numeric_supported(s)
        ]
    else:
        eligible = usable
    rnd = random.Random(seed)
    rnd.shuffle(eligible)
    return eligible[:limit], len(eligible)


def planner_metrics(plan: dict, sample: dict) -> dict:
    gold_sets = collect_gold_answer_sets(sample)
    materials = str(plan.get("materials") or "")
    hits = plan.get("_raw_hits") or []
    hit_text = " ".join(getattr(h, "content", "") for h in hits[:8])
    covered_materials = sum(1 for answers in gold_sets if answer_appears(materials, answers))
    covered_hits = sum(1 for answers in gold_sets if answer_appears(hit_text, answers))
    return {
        "gold_points": len(gold_sets),
        "covered_in_materials": covered_materials,
        "covered_in_hits": covered_hits,
        "materials_coverage": covered_materials / max(len(gold_sets), 1),
        "hits_coverage": covered_hits / max(len(gold_sets), 1),
        "hits_count": int(plan.get("hits_count") or len(hits)),
    }


def make_corrupted_answer(sample: dict, decoys: list[str]) -> tuple[str, dict]:
    answer = str(sample["answer"])
    gold_sets = collect_gold_answer_sets(sample)
    for answers in gold_sets:
        for gold in answers:
            gold_norm = normalize_text(gold)
            if len(gold_norm) < 3:
                continue
            if gold in answer:
                for decoy in decoys:
                    if normalize_text(decoy) != gold_norm and decoy not in answer:
                        corrupted = answer.replace(gold, decoy, 1)
                        return corrupted, {"type": "entity_swap", "from": gold, "to": decoy}
    injection = answer.rstrip() + " It was officially revised in 2019 according to later reports."
    return injection, {"type": "unsupported_injection", "from": "", "to": "2019 later reports"}


def parse_baseline_critic(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {"passed": False, "issues": [{"raw": "<empty-response>"}], "raw": raw, "invalid": True}
    if raw.upper() == "OK":
        return {"passed": True, "issues": [], "raw": raw, "invalid": False}
    issues = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("-") or line.startswith("1.") or line.startswith("2."):
            issues.append({"raw": line})
    if not issues:
        issues = [{"raw": raw[:200]}]
    return {"passed": False, "issues": issues, "raw": raw, "invalid": False}


def run_baseline_critic(
    *,
    content: str,
    outline: str,
    materials: str,
) -> dict:
    """Simple non-LLM baseline: unsupported entity/number spotting.

    This baseline is intentionally lightweight so local runs always finish.
    It serves as a deterministic lower bound for the current LLM-based critic.
    """
    materials_norm = normalize_text(materials)
    issues = []
    sentences = [s.strip() for s in re.split(r"[。！？.!?]\s*", content) if s.strip()]
    for sent in sentences[:12]:
        # Flag sentences with numbers absent from evidence.
        nums = re.findall(r"\b\d[\d,\.:%-]*\b", sent)
        missing_nums = [n for n in nums if n and n not in materials]
        if missing_nums:
            issues.append({
                "raw": f"- unsupported number: {missing_nums[0]} | {sent[:120]}",
            })
            continue

        # Flag when multiple salient tokens do not appear in materials.
        toks = [t for t in tokenize(sent) if len(t) >= 4][:12]
        absent = [t for t in toks if t not in materials_norm]
        if len(absent) >= 4 and len(toks) >= 6:
            issues.append({
                "raw": f"- unsupported claim: {', '.join(absent[:4])} | {sent[:120]}",
            })

    return {
        "passed": len(issues) == 0,
        "issues": issues[:6],
        "raw": "OK" if not issues else "\n".join(i["raw"] for i in issues[:6]),
        "invalid": False,
    }


def choose_samples(all_samples: list[dict], limit: int, seed: int) -> list[dict]:
    usable = [s for s in all_samples if sample_is_usable(s)]
    rnd = random.Random(seed)
    rnd.shuffle(usable)
    return usable[:limit]


def build_decoy_pool(samples: list[dict]) -> list[str]:
    pool = []
    for sample in samples:
        for answers in collect_gold_answer_sets(sample):
            for ans in answers:
                if len(str(ans).strip()) >= 3:
                    pool.append(str(ans).strip())
    # preserve order, remove duplicates
    seen = set()
    out = []
    for item in pool:
        norm = normalize_text(item)
        if norm not in seen:
            seen.add(norm)
            out.append(item)
    return out


def summarize_planner(records: list[dict]) -> dict:
    n = max(len(records), 1)
    return {
        "cases": len(records),
        "avg_gold_points": round(sum(r["gold_points"] for r in records) / n, 3),
        "avg_materials_coverage": round(sum(r["materials_coverage"] for r in records) / n, 4),
        "avg_hits_coverage": round(sum(r["hits_coverage"] for r in records) / n, 4),
        "avg_hits_count": round(sum(r["hits_count"] for r in records) / n, 3),
        "full_materials_coverage_rate": round(sum(1 for r in records if r["materials_coverage"] >= 0.999) / n, 4),
    }


def summarize_critic(records: list[dict], prefix: str) -> dict:
    clean = [r for r in records if r["variant"] == "clean" and r["system"] == prefix]
    corrupt = [r for r in records if r["variant"] == "corrupt" and r["system"] == prefix]
    n_clean = max(len(clean), 1)
    n_corrupt = max(len(corrupt), 1)
    clean_pass = sum(1 for r in clean if r["passed"])
    corrupt_detect = sum(1 for r in corrupt if not r["passed"])
    return {
        "clean_cases": len(clean),
        "corrupt_cases": len(corrupt),
        "clean_pass_rate": round(clean_pass / n_clean, 4),
        "corrupt_detect_rate": round(corrupt_detect / n_corrupt, 4),
        "balanced_score": round((clean_pass / n_clean + corrupt_detect / n_corrupt) / 2, 4),
        "avg_issue_count_clean": round(sum(r["issue_count"] for r in clean) / n_clean, 3),
        "avg_issue_count_corrupt": round(sum(r["issue_count"] for r in corrupt) / n_corrupt, 3),
        "invalid_rate": round(sum(1 for r in clean + corrupt if r["invalid"]) / max(len(clean) + len(corrupt), 1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner/Critic local evaluation on ALCE ASQA")
    parser.add_argument("--model", default="qwen3.5:9b", help="Local Ollama model name")
    parser.add_argument("--endpoint", default="http://localhost:11434", help="Ollama endpoint")
    parser.add_argument("--planner-limit", type=int, default=4, help="Number of planner samples")
    parser.add_argument("--critic-limit", type=int, default=4, help="Number of critic samples")
    parser.add_argument(
        "--critic-subset",
        default="default",
        choices=["default", "strict", "strict_numeric"],
        help="Critic sample selection policy",
    )
    parser.add_argument(
        "--critic-content-chars",
        type=int,
        default=900,
        help="Max characters of answer fed into critic for stable local evaluation",
    )
    parser.add_argument(
        "--planner-use-llm",
        action="store_true",
        help="Use local LLM to generate planner knowledge points instead of gold ASQA qa_pairs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--alce-tar",
        default=str(PROJ / "result" / "alce_data.tar"),
        help="Path to cached ALCE tar file",
    )
    parser.add_argument(
        "--output",
        default=str(PROJ / "result" / "planner_critic_alce_local_result.json"),
        help="JSON output path",
    )
    args = parser.parse_args()

    _set_local_llm_env(args.model, args.endpoint)

    from agent_file_create.document._critic import run_critic
    import agent_file_create.rag.planner as planner_mod
    from agent_file_create.rag.planner import plan_section_knowledge

    # Local test harness: avoid the very slow LLM sentence-filter fallback when
    # FlagEmbedding is unavailable. This keeps the retrieval/planning path
    # under test while making the evaluation practical on a local machine.
    planner_mod._rerank_sentences = fast_local_rerank

    t0 = time.perf_counter()
    alce_tar = ensure_alce_tar(Path(args.alce_tar))
    samples = load_asqa_samples(alce_tar)
    planner_samples = choose_samples(samples, args.planner_limit, args.seed)
    critic_samples, critic_eligible_count = choose_critic_samples(
        samples,
        args.critic_limit,
        args.seed + 1,
        args.critic_subset,
    )
    decoys = build_decoy_pool(samples[:200])

    print(f"Model: {args.model}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Dataset: ALCE / ASQA ({len(samples)} total samples)")
    print(f"Planner cases: {len(planner_samples)}")
    print(f"Critic cases: {len(critic_samples)}")
    print(f"Critic subset: {args.critic_subset} (eligible={critic_eligible_count})")
    print(f"Planner LLM checklist: {'on' if args.planner_use_llm else 'off (use gold qa_pairs)'}")
    print(f"Critic content chars: {args.critic_content_chars}")

    planner_baseline_records = []
    planner_opt_records = []

    print("\n=== Planner Evaluation ===")
    for idx, sample in enumerate(planner_samples, start=1):
        question = str(sample["question"]).strip()
        kb = MiniKB(sample.get("docs") or [], kb_name=f"asqa_{idx}")

        base_plan = baseline_plan(sample, kb)
        base_metrics = planner_metrics(base_plan, sample)
        planner_baseline_records.append({"question": question, **base_metrics})

        if args.planner_use_llm:
            opt_plan = plan_section_knowledge(
                section_title=question,
                parent_title="ALCE ASQA",
                user_prompt=question,
                kb=kb,
                kb_name=kb.kb_name,
                max_points=4,
                target_words=1200,
            )
        else:
            gold_kps = [str(pair.get("question") or "").strip() for pair in (sample.get("qa_pairs") or [])]
            opt_plan = current_plan_fast(
                question=question,
                knowledge_points=[kp for kp in gold_kps if kp][:4],
                kb=kb,
                planner_mod=planner_mod,
                target_words=1200,
            )
        opt_metrics = planner_metrics(opt_plan, sample)
        planner_opt_records.append({"question": question, **opt_metrics})

        print(
            f"[{idx}] {question[:70]}\n"
            f"  baseline: cov(materials)={base_metrics['materials_coverage']:.2f}, "
            f"cov(hits)={base_metrics['hits_coverage']:.2f}, hits={base_metrics['hits_count']}\n"
            f"  current : cov(materials)={opt_metrics['materials_coverage']:.2f}, "
            f"cov(hits)={opt_metrics['hits_coverage']:.2f}, hits={opt_metrics['hits_count']}"
        )

    critic_records = []
    print("\n=== Critic Evaluation ===")
    for idx, sample in enumerate(critic_samples, start=1):
        question = str(sample["question"]).strip()
        materials = build_materials(sample)
        outline = build_outline(sample)
        clean = str(sample["answer"]).strip()[: args.critic_content_chars]
        corrupt, corruption_meta = make_corrupted_answer(sample, decoys)
        corrupt = corrupt[: args.critic_content_chars]

        variants = [("clean", clean), ("corrupt", corrupt)]
        for variant_name, content in variants:
            base = run_baseline_critic(content=content, outline=outline, materials=materials)
            cur = run_critic(content=content, outline=outline, materials=materials)
            cur_invalid = bool(cur.get("error")) or (not cur.get("raw") and len(content) >= 100 and not cur.get("issues"))

            critic_records.append(
                {
                    "question": question,
                    "variant": variant_name,
                    "system": "baseline",
                    "passed": bool(base.get("passed")),
                    "issue_count": len(base.get("issues") or []),
                    "invalid": bool(base.get("invalid")),
                    "corruption": corruption_meta if variant_name == "corrupt" else {},
                }
            )
            critic_records.append(
                {
                    "question": question,
                    "variant": variant_name,
                    "system": "current",
                    "passed": bool(cur.get("passed")),
                    "issue_count": len(cur.get("issues") or []),
                    "invalid": cur_invalid,
                    "error": str(cur.get("error") or ""),
                    "corruption": corruption_meta if variant_name == "corrupt" else {},
                }
            )

        base_clean = critic_records[-4]
        cur_clean = critic_records[-3]
        base_corrupt = critic_records[-2]
        cur_corrupt = critic_records[-1]
        print(
            f"[{idx}] {question[:70]}\n"
            f"  corruption: {corruption_meta}\n"
            f"  baseline clean={'PASS' if base_clean['passed'] else 'FLAG'} / "
            f"corrupt={'DETECT' if not base_corrupt['passed'] else 'MISS'}\n"
            f"  current  clean={'PASS' if cur_clean['passed'] else 'FLAG'} / "
            f"corrupt={'DETECT' if not cur_corrupt['passed'] else 'MISS'}"
        )

    planner_baseline_summary = summarize_planner(planner_baseline_records)
    planner_current_summary = summarize_planner(planner_opt_records)
    critic_baseline_summary = summarize_critic(critic_records, "baseline")
    critic_current_summary = summarize_critic(critic_records, "current")

    planner_delta = {
        "materials_coverage_gain": round(
            planner_current_summary["avg_materials_coverage"] - planner_baseline_summary["avg_materials_coverage"], 4
        ),
        "hits_coverage_gain": round(
            planner_current_summary["avg_hits_coverage"] - planner_baseline_summary["avg_hits_coverage"], 4
        ),
        "avg_hits_count_gain": round(
            planner_current_summary["avg_hits_count"] - planner_baseline_summary["avg_hits_count"], 4
        ),
    }
    critic_delta = {
        "balanced_score_gain": round(
            critic_current_summary["balanced_score"] - critic_baseline_summary["balanced_score"], 4
        ),
        "corrupt_detect_gain": round(
            critic_current_summary["corrupt_detect_rate"] - critic_baseline_summary["corrupt_detect_rate"], 4
        ),
        "clean_pass_gain": round(
            critic_current_summary["clean_pass_rate"] - critic_baseline_summary["clean_pass_rate"], 4
        ),
    }

    elapsed = time.perf_counter() - t0
    result = {
        "config": {
            "model": args.model,
            "endpoint": args.endpoint,
            "planner_limit": args.planner_limit,
            "critic_limit": args.critic_limit,
            "critic_subset": args.critic_subset,
            "seed": args.seed,
            "alce_tar": str(alce_tar),
        },
        "dataset_choice": {
            "benchmark": "ALCE",
            "subset": "ASQA",
            "rationale": [
                "ASQA provides qa_pairs, which map well to planner knowledge decomposition.",
                "ALCE provides answer plus retrieved docs, which map well to critic factual review.",
            ],
        },
        "planner": {
            "baseline_summary": planner_baseline_summary,
            "current_summary": planner_current_summary,
            "delta": planner_delta,
            "per_case": {
                "baseline": planner_baseline_records,
                "current": planner_opt_records,
            },
        },
        "critic": {
            "baseline_summary": critic_baseline_summary,
            "current_summary": critic_current_summary,
            "delta": critic_delta,
            "per_case": critic_records,
        },
        "runtime_seconds": round(elapsed, 3),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"Planner baseline avg materials coverage: {planner_baseline_summary['avg_materials_coverage']:.3f}")
    print(f"Planner current  avg materials coverage: {planner_current_summary['avg_materials_coverage']:.3f}")
    print(f"Planner coverage gain: {planner_delta['materials_coverage_gain']:+.3f}")
    print(f"Critic baseline balanced score: {critic_baseline_summary['balanced_score']:.3f}")
    print(f"Critic current  balanced score: {critic_current_summary['balanced_score']:.3f}")
    print(f"Critic balanced gain: {critic_delta['balanced_score_gain']:+.3f}")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
