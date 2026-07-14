#!/usr/bin/env python3
"""Unified test runner for agent-file-create.

Runs all test phases in dependency order:
  1. Retrieval quality (pure logic, no LLM/DB)
  2. Regex comparison (pure logic, no LLM)
  3. LLM comparison (requires Ollama)
  4. Missing gaps test (requires Ollama)
  5. Planner + Critic end-to-end (requires Ollama)

Usage:
  # Default (Ollama + qwen3.5:9b)
  python tests/run_all.py

  # Custom model
  MODEL=qwen2.5:7b python tests/run_all.py

  # Skip LLM-dependent tests
  SKIP_LLM=1 python tests/run_all.py
"""
import asyncio, os, sys, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from tests.test_utils import print_section_header, STYLE, MODEL


async def run_phase(name, cmd_or_module, is_module=True):
    """Run a test phase with timing."""
    print_section_header(f"Phase: {name}")
    print(f"  Starting at: {time.strftime('%H:%M:%S')}")
    t0 = time.perf_counter()

    try:
        if is_module:
            # Import and run module
            import importlib
            mod = importlib.import_module(cmd_or_module)
            if hasattr(mod, "run"):
                if asyncio.iscoroutinefunction(mod.run):
                    await mod.run()
                else:
                    mod.run()
            elif hasattr(mod, "run_comparison"):
                await mod.run_comparison()
            elif hasattr(mod, "run_eval"):
                mod.run_eval()
            elif hasattr(mod, "run_test"):
                await mod.run_test()
            else:
                print(f"  WARNING: No run()/run_test() found in {cmd_or_module}")
        else:
            # Shell command
            rc = os.system(cmd_or_module)
            if rc != 0:
                print(f"  WARNING: Command returned {rc}")

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    elapsed = time.perf_counter() - t0
    print(f"  Duration: {elapsed:.1f}s")
    return True


async def main():
    print_section_header("Unified Test Runner")
    print(f"  Model: {STYLE}/{MODEL}")
    skip_llm = os.getenv("SKIP_LLM", "") == "1"
    if skip_llm:
        print("  LLM-dependent tests SKIPPED (SKIP_LLM=1)")
    print()

    phases = [
        ("1. 检索质量测试(纯逻辑)", "tests.test_retrieval_quality", True),
        ("2. 正则层对比测试(无LLM)", "tests.test_comparison", True),
    ]

    if not skip_llm:
        phases += [
            ("3. LLM层对比测试", "tests.test_llm_comparison", True),
            ("4. 补充缺口测试", "tests.test_missing", True),
            ("5. Planner+Critic端到端", "tests.test_planner_critic", True),
        ]

    passed = 0
    failed = 0
    total_t0 = time.perf_counter()

    for name, module, is_mod in phases:
        ok = await run_phase(name, module, is_mod)
        if ok:
            passed += 1
        else:
            failed += 1
        print()  # spacing

    total_elapsed = time.perf_counter() - total_t0
    print_section_header("All Phases Complete")
    print(f"  Passed: {passed}/{len(phases)}")
    print(f"  Failed: {failed}/{len(phases)}")
    print(f"  Total time: {total_elapsed:.1f}s")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
