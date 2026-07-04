"""Excel/CSV analysis skill — statistical summary + trend detection.

Reads a data file, computes descriptive statistics, detects trends,
and returns a structured summary suitable for report generation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_file_create.skills.base import SkillResult, SkillMeta, skill

logger = logging.getLogger(__name__)


def _analyze(df, analysis_type: str) -> str:
    """Core analysis: produce a markdown summary of the DataFrame."""
    lines = []
    rows, cols = df.shape
    lines.append(f"数据概览：{rows} 行 × {cols} 列\n")
    lines.append(f"列名：{', '.join(str(c) for c in df.columns)}\n")

    # ── Numeric columns ──
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        lines.append("## 数值列统计\n")
        stats = df[num_cols].describe().round(2)
        for col in num_cols:
            s = stats[col]
            lines.append(f"- **{col}**: 均值={s['mean']}, 中位数={s['50%']}, "
                         f"标准差={s['std']}, 最小={s['min']}, 最大={s['max']}")
        lines.append("")

    # ── Trend detection ──
    if analysis_type in ("trend", "all") and len(num_cols) >= 1:
        lines.append("## 趋势分析\n")
        for col in num_cols[:5]:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            # Simple linear trend: compare first half vs second half
            mid = len(series) // 2
            first_half = series.iloc[:mid].mean()
            second_half = series.iloc[mid:].mean()
            if second_half > first_half * 1.05:
                direction = "上升"
            elif second_half < first_half * 0.95:
                direction = "下降"
            else:
                direction = "平稳"
            change_pct = abs(second_half - first_half) / max(abs(first_half), 0.01) * 100
            lines.append(f"- **{col}**: {direction}趋势，变动幅度约 {change_pct:.1f}% "
                         f"(前半段均值={first_half:.2f}, 后半段均值={second_half:.2f})")
        lines.append("")

    # ── Comparison ──
    if analysis_type in ("comparison", "all") and len(num_cols) >= 2:
        lines.append("## 对比分析\n")
        # Find the column with highest variance for comparison
        variances = [(col, df[col].var()) for col in num_cols if df[col].var() > 0]
        variances.sort(key=lambda x: x[1], reverse=True)
        if variances:
            top_col = variances[0][0]
            for col in num_cols:
                if col == top_col:
                    continue
                corr = df[top_col].corr(df[col])
                if abs(corr) > 0.3:
                    rel = "正相关" if corr > 0 else "负相关"
                    lines.append(f"- {top_col} 与 {col}: {rel} (r={corr:.2f})")
        lines.append("")

    # ── Top/bottom rows ──
    if analysis_type in ("summary", "all") and len(num_cols) >= 1:
        lines.append("## 极值\n")
        for col in num_cols[:3]:
            top3 = df.nlargest(3, col)
            bot3 = df.nsmallest(3, col)
            lines.append(f"- **{col}** 最高值: {', '.join(str(v) for v in top3[col].head(3).tolist())}")
            lines.append(f"  最低值: {', '.join(str(v) for v in bot3[col].head(3).tolist())}")
        lines.append("")

    return "\n".join(lines)


async def _execute_excel_analyze(
    file_path: str = "",
    analysis_type: str = "all",
    **kwargs,
) -> SkillResult:
    """Analyze an Excel/CSV file and return statistical summary."""
    if not file_path:
        return SkillResult(success=False, error="未指定文件路径")

    p = Path(file_path)
    if not p.exists():
        return SkillResult(success=False, error=f"文件不存在: {file_path}")

    try:
        import pandas as pd
    except ImportError:
        return SkillResult(success=False, error="pandas 未安装，请执行 pip install pandas openpyxl")

    try:
        suf = p.suffix.lower()
        if suf in (".xlsx", ".xls"):
            df = pd.read_excel(str(p), engine="openpyxl" if suf == ".xlsx" else None)
        elif suf == ".csv":
            df = pd.read_csv(str(p), encoding="utf-8")
        else:
            return SkillResult(success=False, error=f"不支持的文件格式: {suf}")
    except Exception as exc:
        # Try alternate encoding for CSV
        if suf == ".csv":
            try:
                df = pd.read_csv(str(p), encoding="gbk")
            except Exception:
                return SkillResult(success=False, error=f"读取文件失败: {str(exc)[:200]}")
        else:
            return SkillResult(success=False, error=f"读取文件失败: {str(exc)[:200]}")

    if df.empty:
        return SkillResult(success=False, error="文件为空或无可读取的数据")

    valid_types = {"summary", "trend", "comparison", "all"}
    atype = analysis_type if analysis_type in valid_types else "all"

    try:
        summary = _analyze(df, atype)
    except Exception as exc:
        return SkillResult(success=False, error=f"分析失败: {str(exc)[:200]}")

    # Extract data summary for downstream chart generation
    data_info = {
        "file": str(p.name),
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "sample_values": {
            col: df[col].dropna().head(10).tolist()
            for col in df.select_dtypes(include=["number"]).columns[:3]
        },
    }

    return SkillResult(
        success=True,
        summary=summary,
        data=data_info,
        tokens_used=len(summary) // 3,
    )


SKILL_META = skill(
    name="excel_analyze",
    description="分析Excel/CSV数据文件，提取统计摘要、趋势、相关性，适合数据驱动的报告",
    category="file_processing",
    stage="enrich",
    parameters={
        "file_path": {"type": "string", "description": "Excel/CSV文件的完整路径"},
        "analysis_type": {"type": "string", "description": "分析类型: summary|trend|comparison|all", "default": "all"},
    },
    timeout_s=30,
    max_retries=1,
)(_execute_excel_analyze)
