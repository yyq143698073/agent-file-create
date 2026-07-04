"""Chart generation skill — create matplotlib/echarts charts for reports.

Generates bar, line, pie, or scatter charts from Excel/CSV data or
inline specification. Saves PNG files and returns markdown image refs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_file_create.skills.base import SkillResult, SkillMeta, skill

logger = logging.getLogger(__name__)

# Color palette optimized for Chinese reports (clear, professional)
_COLORS = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5",
           "#70AD47", "#264478", "#9B59B6", "#E74C3C", "#1ABC9C"]

# Enable Chinese font support
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    # Try to set Chinese-capable font
    _CN_FONTS = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei",
                 "Noto Sans CJK SC", "PingFang SC", "STHeiti", "sans-serif"]
    for _f in _CN_FONTS:
        try:
            plt.rcParams["font.sans-serif"] = [_f]
            # Quick validation: render a single Chinese character
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.set_title("测")
            plt.close(fig)
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False


def _load_data(file_path: str) -> "pd.DataFrame | None":
    """Load data from Excel/CSV, returning None on failure."""
    try:
        import pandas as pd
    except ImportError:
        return None
    p = Path(file_path)
    if not p.exists():
        return None
    try:
        suf = p.suffix.lower()
        if suf in (".xlsx", ".xls"):
            return pd.read_excel(str(p), engine="openpyxl" if suf == ".xlsx" else None)
        elif suf == ".csv":
            try:
                return pd.read_csv(str(p), encoding="utf-8")
            except Exception:
                return pd.read_csv(str(p), encoding="gbk")
    except Exception:
        return None


def _make_bar(df, x_col: str, y_col: str, title: str, output_path: str) -> str:
    """Generate a bar chart. Returns description string."""
    x = df[x_col].astype(str).tolist()
    y = df[y_col].astype(float).tolist()
    if len(x) > 20:
        x = x[:20]
        y = y[:20]
        title += " (前20项)"

    fig, ax = plt.subplots(figsize=(max(8, len(x) * 0.5), 5))
    bars = ax.bar(range(len(x)), y, color=_COLORS[0], edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on bars
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(y) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"柱状图({len(x)}项)"


def _make_line(df, x_col: str, y_cols: list[str], title: str, output_path: str) -> str:
    """Generate a line chart (supports multiple y columns)."""
    x = df[x_col].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(x) * 0.4), 5))
    for i, y_col in enumerate(y_cols[:4]):
        y = df[y_col].astype(float).tolist()
        ax.plot(range(len(x)), y, marker="o", markersize=3, linewidth=1.5,
                color=_COLORS[i % len(_COLORS)], label=y_col)
    ax.set_xticks(range(0, len(x), max(1, len(x) // 10)))
    ax.set_xticklabels([x[i] for i in range(0, len(x), max(1, len(x) // 10))],
                       rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"折线图({len(x)}点, {len(y_cols)}条线)"


def _make_pie(df, label_col: str, value_col: str, title: str, output_path: str) -> str:
    """Generate a pie chart (≤8 slices to stay readable)."""
    df_sorted = df.nlargest(7, value_col)
    labels = df_sorted[label_col].astype(str).tolist()
    values = df_sorted[value_col].astype(float).tolist()
    # Group remainder as "其他"
    other_val = df[value_col].sum() - df_sorted[value_col].sum()
    if other_val > 0:
        labels.append("其他")
        values.append(other_val)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=_COLORS[:len(labels)],
        startangle=90, pctdistance=0.6,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"饼图({len(labels)}类)"


def _make_scatter(df, x_col: str, y_col: str, title: str, output_path: str) -> str:
    """Generate a scatter plot."""
    x = df[x_col].astype(float).tolist()
    y = df[y_col].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, c=_COLORS[0], alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"散点图({len(x)}个点)"


_CHART_FNS = {
    "bar": _make_bar,
    "line": _make_line,
    "pie": _make_pie,
    "scatter": _make_scatter,
}


async def _execute_chart_generate(
    chart_type: str = "bar",
    title: str = "数据图表",
    file_path: str = "",
    x_column: str = "",
    y_column: str = "",
    y_columns: str = "",       # comma-separated for multi-line charts
    label_column: str = "",
    value_column: str = "",
    output_dir: str = "",
    **kwargs,
) -> SkillResult:
    """Generate a chart from data and return the image path + description."""

    if not _MATPLOTLIB_OK:
        return SkillResult(success=False, error="matplotlib 未安装，请执行 pip install matplotlib")

    chart_type = chart_type.strip().lower()
    if chart_type not in _CHART_FNS:
        return SkillResult(
            success=False,
            error=f"不支持的图表类型: {chart_type}，可选: {', '.join(_CHART_FNS)}",
        )

    # Load data
    if not file_path:
        return SkillResult(success=False, error="未指定数据文件路径 (file_path)")
    df = _load_data(file_path)
    if df is None or df.empty:
        return SkillResult(success=False, error=f"无法读取数据: {file_path}")

    # Determine output directory
    out_dir = Path(output_dir) if output_dir else Path(file_path).parent / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = title.replace("/", "_").replace("\\", "_").replace(" ", "_")[:50]
    output_path = str(out_dir / f"{safe_title}_{chart_type}.png")

    import matplotlib.pyplot as plt

    try:
        if chart_type == "bar":
            if not x_column or not y_column:
                return SkillResult(success=False, error="柱状图需要 x_column 和 y_column")
            if x_column not in df.columns or y_column not in df.columns:
                return SkillResult(success=False, error=f"列不存在，可用列: {list(df.columns)}")
            desc = _make_bar(df, x_column, y_column, title, output_path)

        elif chart_type == "line":
            if not x_column:
                return SkillResult(success=False, error="折线图需要 x_column")
            y_cols = [c.strip() for c in (y_columns or y_column).split(",") if c.strip()]
            if not y_cols:
                return SkillResult(success=False, error="折线图需要 y_columns 或 y_column")
            missing = [c for c in y_cols if c not in df.columns]
            if missing:
                return SkillResult(success=False, error=f"列不存在: {missing}，可用: {list(df.columns)}")
            desc = _make_line(df, x_column, y_cols, title, output_path)

        elif chart_type == "pie":
            if not label_column or not value_column:
                return SkillResult(success=False, error="饼图需要 label_column 和 value_column")
            if label_column not in df.columns or value_column not in df.columns:
                return SkillResult(success=False, error=f"列不存在，可用列: {list(df.columns)}")
            desc = _make_pie(df, label_column, value_column, title, output_path)

        elif chart_type == "scatter":
            if not x_column or not y_column:
                return SkillResult(success=False, error="散点图需要 x_column 和 y_column")
            if x_column not in df.columns or y_column not in df.columns:
                return SkillResult(success=False, error=f"列不存在，可用列: {list(df.columns)}")
            desc = _make_scatter(df, x_column, y_column, title, output_path)

        else:
            return SkillResult(success=False, error=f"未知图表类型: {chart_type}")

    except Exception as exc:
        return SkillResult(success=False, error=f"图表生成失败: {str(exc)[:200]}")

    # Build markdown image reference for the report
    img_ref = f"![{title}]({output_path})"
    summary = (
        f"已生成图表: {desc}\n"
        f"标题: {title}\n"
        f"文件: {output_path}\n"
        f"报告中可使用: {img_ref}"
    )

    return SkillResult(
        success=True,
        summary=summary,
        data={
            "chart_type": chart_type,
            "title": title,
            "output_path": output_path,
            "markdown_ref": img_ref,
            "columns_used": [x_column, y_column or y_columns, label_column, value_column],
        },
    )


SKILL_META = skill(
    name="chart_generate",
    description="生成数据图表（柱状图/折线图/饼图/散点图），输出PNG图片，适合报告中的数据可视化",
    category="generation",
    stage="research",
    parameters={
        "chart_type": {"type": "string", "description": "图表类型: bar|line|pie|scatter"},
        "title": {"type": "string", "description": "图表标题"},
        "file_path": {"type": "string", "description": "数据文件路径（Excel/CSV）"},
        "x_column": {"type": "string", "description": "X轴/X列的列名"},
        "y_column": {"type": "string", "description": "Y轴/Y列的列名"},
        "y_columns": {"type": "string", "description": "折线图多线: 逗号分隔的多个Y列名"},
        "label_column": {"type": "string", "description": "饼图: 标签列名"},
        "value_column": {"type": "string", "description": "饼图: 数值列名"},
        "output_dir": {"type": "string", "description": "图片输出目录"},
    },
    timeout_s=30,
    max_retries=1,
)(_execute_chart_generate)
