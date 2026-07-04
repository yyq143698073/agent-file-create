"""Web search skill — searches the internet for up-to-date information.

Uses DuckDuckGo instant answers (no API key required) with a fallback
to a Bing-style web search snippet extraction.
"""

from __future__ import annotations

import logging
from urllib.parse import quote_plus

from agent_file_create.skills.base import SkillResult, SkillMeta, skill

logger = logging.getLogger(__name__)


async def _search_duckduckgo(query: str, max_results: int) -> tuple[str, list[dict]]:
    """Search DuckDuckGo HTML (no API key needed). Returns (summary, articles)."""
    import urllib.request
    import urllib.error

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return "", []

    # Extract result snippets from DuckDuckGo HTML
    import re
    articles: list[dict] = []
    # Each result is in a div with class "result"
    result_blocks = re.findall(
        r'class="result__body".*?</div>\s*</div>',
        html, re.DOTALL
    )

    for block in result_blocks[:max_results]:
        # Extract title
        title_m = re.search(r'class="result__a"[^>]*>([^<]+)<', block)
        title = title_m.group(1).strip() if title_m else ""

        # Extract snippet
        snip_m = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)
        snippet = snip_m.group(1).strip() if snip_m else ""
        snippet = re.sub(r'<[^>]+>', '', snippet)

        # Extract URL
        url_m = re.search(r'class="result__url"[^>]*>(.*?)</a>', block)
        link = url_m.group(1).strip() if url_m else ""

        if title or snippet:
            articles.append({"title": title, "snippet": snippet, "url": link})

    if not articles:
        return "(未找到搜索结果)", []

    # Build summary
    lines = [f"搜索关键词: {query}", f"找到 {len(articles)} 条结果:\n"]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a['title']}")
        if a['snippet']:
            lines.append(f"   {a['snippet'][:300]}")
        if a['url']:
            lines.append(f"   来源: {a['url']}")
        lines.append("")

    return "\n".join(lines), articles


async def _execute_web_search(query: str, max_results: int = 5, **kwargs) -> SkillResult:
    """Execute web search."""
    if not query or not query.strip():
        return SkillResult(success=False, error="搜索关键词为空")

    summary, articles = await _search_duckduckgo(query, max_results)

    if not articles:
        return SkillResult(
            success=True,
            summary=f"搜索「{query}」未找到结果，建议更换关键词。",
            data={"query": query, "articles": [], "count": 0},
        )

    return SkillResult(
        success=True,
        summary=summary,
        data={"query": query, "articles": articles, "count": len(articles)},
        tokens_used=len(summary) // 3,
    )


SKILL_META = skill(
    name="web_search",
    description="搜索互联网获取最新信息、新闻、数据，适合需要时效性内容的报告",
    category="research",
    stage="research",
    parameters={
        "query": {"type": "string", "description": "搜索关键词"},
        "max_results": {"type": "integer", "description": "返回结果数量", "default": 5},
    },
    timeout_s=30,
    max_retries=1,
)(_execute_web_search)
