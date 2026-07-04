"""Citation verification step — check that cited claims match cited sources."""

import logging
import re
from difflib import SequenceMatcher as _SM

from agent_file_create.quality.step import QualityContext, QualityStep, StepResult

logger = logging.getLogger(__name__)


class CitationStep(QualityStep):
    """Verify that in-text citations (据...) reference real source files.

    Uses fuzzy matching with fallback word-level comparison.
    Auto-fills placeholder citations ("据同一材料", "据文献", etc.)
    using the last successfully matched citation.

    Also detects vague/unsupported citations (据报、有研究显示) that
    cannot be traced to any specific source file.
    """

    name = "citation"

    _PLACEHOLDER_PATTERNS = [
        "同一材料", "同份材料", "同研究", "同一研究",
        "同上", "同文献", "同来源", "据材料显示",
        "据资料记载", "据文献", "据实验数据",
    ]

    # Vague citation patterns — claims that cite unnamed sources
    _VAGUE_CITATION_PATTERNS = [
        (re.compile(r"据[^，。；]{0,20}(?:报告|统计|数据|调查|预测|分析)"), "未指明具体报告名称"),
        (re.compile(r"(?:有研究|有学者|有报告|有文章)[^，。；]{0,10}(?:表明|显示|指出|发现|认为)"), "未指明研究/学者名称"),
        (re.compile(r"据[报悉了][^，。；]{0,8}[道称]"), "未指明信息来源"),
        (re.compile(r"(?:据了解|据悉|据闻|据报道|据透露)"), "使用了不确定来源的引述"),
        (re.compile(r"(?:相关研究|相关报告|相关数据|相关调查)[^，。；]{0,10}(?:表明|显示|指出)"), "未指明具体出处"),
    ]

    def run(self, ctx: QualityContext) -> StepResult:
        content = ctx.content
        analysis_results = ctx.analysis_results or []
        output_dir = ctx.output_dir

        try:
            _raw_content = str(content or "")
            _citations = re.findall(r"[（(]据(.+?)[）)]", _raw_content)
            if not _citations or not analysis_results:
                return StepResult(success=True, data={"citations_found": 0, "bad": []})

            # Build source map from analysis results
            _source_map = {}
            for _ar in analysis_results:
                _fn = str(_ar.get("filename") or _ar.get("title") or "").strip()
                _summary = str(_ar.get("summary") or "").strip()
                if _fn:
                    _source_map[_fn] = _summary

            _bad_cites: list = []
            _last_good_cite = ""

            for _cite in _citations:
                _cite = _cite.strip()

                # Auto-fill placeholder citations
                _is_placeholder = any(p in _cite for p in self._PLACEHOLDER_PATTERNS)
                if _is_placeholder and _last_good_cite:
                    _old = f"（据{_cite}）"
                    _new_cite = f"（据{_last_good_cite}）"
                    if _old in _raw_content:
                        _raw_content = _raw_content.replace(_old, _new_cite, 1)
                        _cite = _last_good_cite
                        logger.info("citation_autofill %r → %r", _cite, _last_good_cite)
                    _old2 = f"(据{_cite})"
                    if _old2 in _raw_content:
                        _raw_content = _raw_content.replace(_old2, f"(据{_last_good_cite})", 1)

                # Find surrounding context
                _idx = _raw_content.find(f"（据{_cite}）")
                if _idx < 0:
                    _idx = _raw_content.find(f"(据{_cite})")
                _context = ""
                if _idx >= 0:
                    _start = max(0, _idx - 80)
                    _end = min(len(_raw_content), _idx + len(_cite) + 80)
                    _context = _raw_content[_start:_end].replace("\n", " ")

                # Match: exact → fuzzy → word-level
                _best_match = None
                _best_score = 0.0
                for _fn in _source_map:
                    if _cite in _fn or _fn in _cite:
                        _best_match = _fn
                        _best_score = 1.0
                        break
                    _s = _SM(None, _cite, _fn).ratio()
                    if _s > _best_score:
                        _best_score = _s
                        _best_match = _fn
                    if _best_score < 0.5:
                        _cite_words = _cite.replace("、", " ").replace("，", " ").split()
                        if any(w in _fn for w in _cite_words if len(w) >= 2):
                            _best_match = _fn
                            break

                if _best_match and _best_score >= 0.35:
                    _last_good_cite = _cite
                else:
                    _bad_cites.append((_cite, _context))

            # Write back auto-filled content
            if _raw_content != str(content or ""):
                content = _raw_content
                try:
                    from pathlib import Path
                    (Path(output_dir) / "content.md").write_text(content, encoding="utf-8")
                except Exception as e:
                    logger.debug("citation autofill write failed: %s", e)

            if _bad_cites:
                logger.info("citation_verify bad=%d total=%d", len(_bad_cites), len(_citations))
            else:
                logger.info("citation_verify all_ok count=%d", len(_citations))

            # ── Vague citation detection ──────────────────────────────────
            _vague_cites: list[dict] = []
            for _pattern, _reason in self._VAGUE_CITATION_PATTERNS:
                for _match in _pattern.finditer(_raw_content):
                    _matched_text = _match.group(0)
                    # Check if this is inside a specific citation (parenthesized)
                    _start = _match.start()
                    _before = _raw_content[max(0, _start - 30):_start]
                    if "（据" in _before or "(据" in _before:
                        continue  # Already parenthesized — handled above
                    _vague_cites.append({
                        "text": _matched_text[:80],
                        "reason": _reason,
                        "position": _start,
                    })

            # Deduplicate by matched text
            _seen_texts: set[str] = set()
            _unique_vague: list[dict] = []
            for _vc in _vague_cites:
                if _vc["text"] not in _seen_texts:
                    _seen_texts.add(_vc["text"])
                    _unique_vague.append(_vc)

            _vague_warnings: list[str] = []
            for _vc in _unique_vague[:10]:
                _w = f"模糊引用「{_vc['text']}」— {_vc['reason']}"
                _vague_warnings.append(_w)

            if _vague_warnings:
                logger.info("citation_vague_detected count=%d", len(_vague_warnings))

            return StepResult(
                success=True, content=content,
                data={
                    "citations_found": len(_citations),
                    "bad": [{"cite": bc[0], "context": bc[1]} for bc in _bad_cites],
                    "bad_count": len(_bad_cites),
                    "vague_citations": _unique_vague,
                    "vague_count": len(_unique_vague),
                },
                warnings=_vague_warnings,
            )

        except Exception as _e:
            logger.warning("citation_verify_failed err=%s", str(_e)[:200])
            return StepResult(success=False, error=str(_e))
