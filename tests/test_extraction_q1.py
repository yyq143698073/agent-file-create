from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_file_create.preprocessor import (
    choose_better_extraction,
    compute_quality_metrics,
    deduplicate_analysis_results,
)


def test_compute_quality_metrics_flags_missing_required_fields():
    result = {
        "title": "测试标题",
        "keywords": ["测试"],
        "summary": "",
        "key_points": [],
        "data": [],
        "conclusion": "",
        "prediction": "",
    }
    metrics = compute_quality_metrics(result)
    assert metrics["required_ok"] is False
    assert set(metrics["missing_required"]) == {"summary", "key_points"}


def test_choose_better_extraction_prefers_required_fields():
    weak = {
        "title": "报告A",
        "keywords": ["A"],
        "summary": "",
        "key_points": [],
        "data": [],
        "conclusion": "",
        "prediction": "",
    }
    strong = {
        "title": "报告A",
        "keywords": ["A"],
        "summary": "这是一个完整摘要，覆盖了主要发现和背景信息。",
        "key_points": ["发现一", "发现二", "发现三"],
        "data": [{"x": 1}],
        "conclusion": "结论明确",
        "prediction": "",
    }
    chosen = choose_better_extraction(weak, strong)
    assert chosen is strong


def test_deduplicate_analysis_results_merges_similar_sources():
    items = [
        {
            "_file": "a.pdf",
            "title": "新能源汽车市场报告",
            "summary": "2024年新能源汽车销量增长35%，华东地区增长最快。",
            "key_points": ["销量增长35%", "华东增长最快"],
            "data": [{"销量增长": "35%"}],
            "conclusion": "市场继续扩张",
            "prediction": "明年保持增长",
        },
        {
            "_file": "b.pdf",
            "title": "新能源汽车市场报告",
            "summary": "2024年新能源汽车销量增长35%，华东地区增速领先。",
            "key_points": ["销量增长35%", "华东增速领先"],
            "data": [],
            "conclusion": "",
            "prediction": "",
        },
        {
            "_file": "c.pdf",
            "title": "电池技术路线",
            "summary": "固态电池量产时间预计在2027年前后。",
            "key_points": ["固态电池", "2027量产"],
            "data": [],
            "conclusion": "技术仍在推进",
            "prediction": "",
        },
    ]

    deduped = deduplicate_analysis_results(items, threshold=0.75)

    assert len(deduped) == 2
    merged = next(item for item in deduped if item.get("title") == "新能源汽车市场报告")
    assert sorted(merged["_merged_files"]) == ["a.pdf", "b.pdf"]
    assert len(merged["key_points"]) >= 2
