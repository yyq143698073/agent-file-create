# -*- coding: utf-8 -*-
"""
Shared test utilities for the agent-file-create test suite.

Provides:
- Common config loading (STYLE, MODEL, ENDPOINT, KEY)
- Model initialization with consistent defaults
- Multi-sampling runner for non-deterministic LLM tests
- Helper metrics (number extraction, citation counting, claim counting)
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_file_create.llm_factory import get_chat_model


# ── Config ──────────────────────────────────────────────────────────

STYLE   = os.getenv("STYLE",   "ollama")
MODEL   = os.getenv("MODEL",   "qwen3.5:9b")
ENDPOINT = os.getenv("ENDPOINT", "http://localhost:11434")
KEY     = os.getenv("KEY",     "")


# ── Model factory ───────────────────────────────────────────────────

def make_llm(
    temperature: float = 0.01,
    max_tokens: int = 800,
    timeout_s: int = 180,
):
    """Create a chat model with test-common defaults."""
    return get_chat_model(
        style=STYLE, model=MODEL, endpoint=ENDPOINT, api_key=KEY,
        temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
    )


# ── Multi-sampling runner ──────────────────────────────────────────

async def multi_sample_llm(
    runner_fn,
    *,
    n_samples: int = 3,
    timeout_per_sample: int = 300,
    label: str = "",
) -> dict:
    """Run an async LLM test multiple times and aggregate results.

    ``runner_fn`` receives a runner_context dict with keys ``sample_index``
    and ``label``.  Returns a dict with raw results + majority verdict.
    """
    results = []
    errors = []
    timings = []

    for i in range(n_samples):
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                runner_fn({"sample_index": i, "label": label}),
                timeout=timeout_per_sample,
            )
            elapsed = time.perf_counter() - t0
            results.append(result)
            timings.append(elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            errors.append(e)
            timings.append(elapsed)
            results.append(None)

    passed = sum(1 for r in results if r and r.get("passed"))
    failed = len(results) - passed
    overall = passed > failed

    return {
        "label": label or "unnamed",
        "n_samples": n_samples,
        "results": results,
        "errors": errors,
        "timings": timings,
        "passed": passed,
        "failed": failed,
        "overall_passed": overall,
        "avg_time": sum(timings) / len(timings) if timings else 0,
    }


# ── Helper metrics ─────────────────────────────────────────────────

def count_numbers(text: str) -> int:
    return len(re.findall(
        r"\d+(?:\.\d+)?\s*[万亿千百]?\s*(?:%|‰|％|个|项|指标|维度|轮|层|[BbKkMm])?",
        text,
    ))


def count_citations(text: str) -> int:
    return len(re.findall(r"【\d+】", text))


def count_placeholder(text: str) -> int:
    return text.count("需补充数据") + text.count("数据待核实")


def count_claims(text: str) -> int:
    claims = 0
    for sent in re.split(r"[。！？!?\n]", text):
        sent = sent.strip()
        if len(sent) >= 10 and re.search(r"\d", sent):
            claims += 1
    return claims


def extract_all_numbers(text: str) -> set[str]:
    nums = set()
    for m in re.finditer(
        r"\d+(?:\.\d+)?\s*[万亿千百]?\s*(?:%|‰|％|个|项|指标|维度|轮|层|[BbKkMm])?",
        text,
    ):
        v = m.group().strip()
        if len(v) >= 2:
            nums.add(v)
    return nums


def count_unsupported_claims(text: str, source_facts: set[str]) -> int:
    count = 0
    numbers_in_text = extract_all_numbers(text)
    for v in numbers_in_text:
        matched = any(v in f or f in v for f in source_facts)
        if not matched:
            count += 1
    return count


def has_year_citation(text: str) -> bool:
    return bool(re.search(r"(?:20\d{2}|19\d{2})年", text))


def prefers_newer(text: str, newer_year: str = "2024", older_year: str = "2018") -> bool:
    idx_newer = text.find(newer_year)
    idx_older = text.find(older_year)
    if idx_newer >= 0 and idx_older >= 0:
        return idx_newer < idx_older
    return idx_newer >= 0


def prefers_newer_value(text: str, newer_val: str = "82%", older_val: str = "45-55%") -> bool:
    idx_newer_val = text.find(newer_val)
    idx_older_val = text.find(older_val)
    if idx_newer_val >= 0 and idx_older_val >= 0:
        return idx_newer_val < idx_older_val
    return idx_newer_val >= 0


# ── Output formatting ──────────────────────────────────────────────

def print_section_header(title: str, width: int = 60):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_metrics(label: str, value: Any, width: int = 20):
    print(f"  {label:<{width}} {value}")


# ── Assertion helpers ──────────────────────────────────────────────

def summarize_multi_sample(sample_result: dict, test_name: str) -> dict:
    """Print a human-readable summary of a multi-sample test run."""
    passed = sample_result["passed"]
    failed = sample_result["failed"]
    avg_time = sample_result["avg_time"]
    status = "PASS" if sample_result["overall_passed"] else "FAIL"

    print(f"\n  [{test_name}] {sample_result['label']}: "
          f"{status} ({passed}/{failed}/{sample_result['n_samples']} "
          f"pass/fail/total, avg {avg_time:.1f}s)")

    for i, (r, t, e) in enumerate(zip(
        sample_result["results"],
        sample_result["timings"],
        sample_result["errors"],
    )):
        if e:
            print(f"    sample {i+1}: ERROR ({t:.1f}s) — {e}")
        elif r:
            detail = r.get("detail", "")
            print(f"    sample {i+1}: {'PASS' if r.get('passed') else 'FAIL'} "
                  f"({t:.1f}s) {detail}")
        else:
            print(f"    sample {i+1}: FAIL ({t:.1f}s)")

    return sample_result
