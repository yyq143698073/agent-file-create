"""Faithfulness check step — LLM review + hallucination-triggered re-retrieval.

Adds per-issue severity classification and suggested remedies for each finding.
"""

import logging
import os
import re

from agent_file_create.quality.step import QualityContext, QualityStep, StepResult

logger = logging.getLogger(__name__)

# Patterns for severity classification based on issue description keywords
_SEVERITY_PATTERNS: list[tuple[str, str]] = [
    (r"(虚假|编造|伪造|完全不?存在|严重错误|数据错误|虚构)", "严重"),
    (r"(无依据|找不到|未在.*中[发出]现|无来源|缺乏证据|证据不足)", "严重"),
    (r"(推测|推断|可能不?准确|疑似|待核实|需确认|无法验证)", "轻微"),
    (r"(夸大|过度|过于绝对|不够精确|模糊|笼统)", "轻微"),
    (r"(部分.*错误|细节.*问题|表述.*不准确|轻微.*偏差)", "轻微"),
]

# Remedy templates based on issue type
_REMEDY_TEMPLATES: dict[str, str] = {
    "数据": "用源材料中的具体数据替换当前表述，或标注为「估算值」",
    "推测": "添加限定词（如「可能」「据分析」），或引用源材料中的相关观点作为支撑",
    "夸大": "使用源材料中的原始措辞，避免程度副词（如「显著」「大幅」）",
    "缺失": "从知识库检索补充该部分内容，或标注为「待补充」",
    "错误": "直接更正为源材料中的准确信息",
}


def _classify_severity(description: str) -> str:
    """Classify issue severity based on keyword patterns in the description."""
    desc = description or ""
    for pattern, severity in _SEVERITY_PATTERNS:
        if re.search(pattern, desc):
            return severity
    return "中"


def _suggest_remedy(description: str) -> str:
    """Suggest a remedy based on issue type keywords."""
    desc = description or ""
    for keyword, remedy in _REMEDY_TEMPLATES.items():
        if keyword in desc:
            return remedy
    return "对照源材料修改或删除无依据内容"


class FaithfulnessStep(QualityStep):
    """Check content faithfulness against source materials.

    If issues are found, attempts hallucination-triggered re-retrieval
    from the knowledge base to fix flagged sections.

    Each issue is classified by severity (严重/中/轻微) and includes
    a suggested remedy to guide auto-fix or manual correction.
    """

    name = "faithfulness"

    def run(self, ctx: QualityContext) -> StepResult:
        from agent_file_create.llm_client import call_llm as _f_call

        content = ctx.content
        analysis_results = ctx.analysis_results or []
        output_dir = ctx.output_dir
        task_id = ctx.task_id

        try:
            source_text = ""
            for _i, _ar in enumerate(analysis_results[:8]):
                _title = str(_ar.get("title") or _ar.get("filename") or "").strip()
                _summary = str(_ar.get("summary") or "").strip()
                if _title or _summary:
                    source_text += f"[{_i+1}] {_title}: {_summary[:300]}\n"

            if not source_text:
                logger.warning("faithfulness_check skip task=%s reason=no_source_text", task_id)
                if content:
                    _header = (
                        "> ⚠️ **事实核查无法执行**：未从上传材料中提取到足够内容，"
                        "无法验证以下报告的事实准确性。所有数据和结论请以原始材料为准。\n\n"
                    )
                    content = _header + str(content or "")
                    try:
                        from pathlib import Path
                        (Path(output_dir) / "content.md").write_text(content, encoding="utf-8")
                    except Exception as e:
                        logger.debug("faithfulness header write failed: %s", e)
                return StepResult(success=True, content=content,
                                  data={"checked": False, "reason": "no_source_text"})

            if not content:
                logger.warning("faithfulness_check skip task=%s reason=no_content", task_id)
                return StepResult(success=True, content=content,
                                  data={"checked": False, "reason": "no_content"})

            # Phase 1: LLM-based faithfulness check
            _check_prompt = (
                "你是文档事实核查助手。请检查以下报告正文的每个 ## 章节，判断其内容是否能在来源材料中找到依据。\n\n"
                f"来源材料摘要：\n{source_text[:3000]}\n\n"
                f"报告正文：\n{str(content)[:4000]}\n\n"
                "对于每个 ## 章节，判断其可信度：如果章节中的所有事实都能在来源材料中找到明确依据，标记OK；"
                "如果章节中有部分推断或数据无法在来源材料中验证，标记 WARN: <原因>。\n"
                "只输出有问题的章节（OK的不用输出），如果没有问题，回复ALL_OK。\n"
                "格式：## 章节名\nWARN: 原因（简短一句）"
            )
            _check_result = _f_call(
                _check_prompt, timeout_s=20, temperature=0.0, num_predict=300,
                system="你是一个中文文档处理助手。只输出有问题的章节。")
            _check_text = (_check_result or "").strip()

            if not _check_text:
                logger.warning("faithfulness_check empty_response task=%s", task_id)
                return StepResult(success=True, content=content, data={"checked": True, "warnings": []})

            if _check_text.upper() in {"ALL_OK", "OK", "无", "NONE"}:
                logger.info("faithfulness_check all_ok task=%s", task_id)
                return StepResult(success=True, content=content, data={"checked": True, "warnings": []})

            # Phase 2: Parse warnings with severity and remedy
            _warnings: list = []
            _cur_section = ""
            for _line in _check_text.splitlines():
                _line = _line.strip()
                if _line.startswith("## "):
                    _cur_section = _line[3:].strip()
                elif (_line.startswith("WARN:") or _line.startswith("WARN：")) and _cur_section:
                    _reason = _line[5:].strip().lstrip(":：").strip()
                    _severity = _classify_severity(_reason)
                    _remedy = _suggest_remedy(_reason)
                    _warnings.append({
                        "section": _cur_section,
                        "reason": _reason,
                        "severity": _severity,
                        "remedy": _remedy,
                    })
                    _cur_section = ""

            if not _warnings:
                return StepResult(success=True, content=content, data={"checked": True, "warnings": []})

            # Phase 3: Hallucination-triggered re-retrieval
            _enable_retrieval = os.getenv("HALLUCINATION_RETRIEVAL_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
            _fixed_count = 0
            _unfixed: list = []

            try:
                from agent_file_create.rag.kb import KnowledgeBase
                _kb = KnowledgeBase() if _enable_retrieval else None
                _kb_name = str(task_id)
            except Exception:
                _kb = None
                _kb_name = ""

            annotated = str(content or "")
            for _w in _warnings:
                _sec_name = _w["section"]
                _reason = _w["reason"]
                _did_fix = False
                if _kb:
                    try:
                        _sec_body = ""
                        _in_sec = False
                        for _cl in annotated.splitlines():
                            if _cl.strip().startswith(f"## {_sec_name}"):
                                _in_sec = True
                                continue
                            if _in_sec:
                                if _cl.strip().startswith("## "):
                                    break
                                if not _cl.strip().startswith("> ⚠️"):
                                    _sec_body += _cl + "\n"
                        _sec_body = _sec_body.strip()

                        if _sec_body and len(_sec_body) > 50:
                            _query_prompt = (
                                "你是一个搜索查询生成助手。以下段落的事实核查未通过，需要从知识库中检索正确的信息来验证。\n"
                                "请针对段落中涉及的每个关键主题，生成1-3个中性的搜索查询（不要重复段落的原话，\n"
                                "而是提取核心主题和实体，用不同的措辞表达）。\n\n"
                                f"段落：{_sec_body[:800]}\n\n"
                                "每行一个搜索查询，只输出查询本身。"
                            )
                            _queries_raw = _f_call(
                                _query_prompt, timeout_s=12, temperature=0.0, num_predict=200,
                                system="你只输出搜索查询，每行一个。")
                            _queries = [q.strip() for q in (_queries_raw or "").splitlines()
                                        if q.strip() and len(q.strip()) > 3][:4]

                            _claim_prompt = (
                                "从以下段落中提取1-2个需要验证的关键事实主张。每行一个，只输出主张本身。\n\n"
                                f"段落：{_sec_body[:800]}"
                            )
                            _claims_raw = _f_call(
                                _claim_prompt, timeout_s=10, temperature=0.0, num_predict=150,
                                system="你只输出关键主张，每行一个。")
                            _claims = [c.strip() for c in (_claims_raw or "").splitlines()
                                       if c.strip() and len(c.strip()) > 5][:3]

                            _all_queries = _queries + _claims

                            if _all_queries:
                                _new_hits = []
                                _seen = set()
                                for _q in _all_queries:
                                    try:
                                        _h = _kb.search_hyde(kb=_kb_name, query=_q, top_k=3)
                                        for _hit in _h:
                                            if _hit.chunk_id not in _seen:
                                                _seen.add(_hit.chunk_id)
                                                _new_hits.append(_hit)
                                    except Exception:
                                        try:
                                            _h = _kb.search_adaptive(kb=_kb_name, query=_q, top_k=3)
                                            for _hit in _h:
                                                if _hit.chunk_id not in _seen:
                                                    _seen.add(_hit.chunk_id)
                                                    _new_hits.append(_hit)
                                        except Exception:
                                            pass

                                if _new_hits:
                                    _new_ctx = ""
                                    for _j, _h in enumerate(_new_hits[:5]):
                                        _chunk = str(_h.content or "")[:400]
                                        if _chunk:
                                            _new_ctx += f"[新检索{_j+1}] {_chunk}\n"

                                    if _new_ctx:
                                        _fix_prompt = (
                                            "你是一个文档编辑助手。以下段落的事实核查未通过，"
                                            "因为部分内容在原始材料中找不到依据。请用新增检索到的材料重写该段落，"
                                            "只保留有材料支撑的内容。如果新材料能支撑原主张，保留它；"
                                            "如果不能，用新材料替换或删除无支撑的内容。\n\n"
                                            f"原始段落（有问题）：\n{_sec_body[:600]}\n\n"
                                            f"新增检索材料：\n{_new_ctx[:1200]}\n\n"
                                            "请输出修正后的段落（只输出正文，不使用HTML注释、不使用核查标记）："
                                        )
                                        _fixed_body = _f_call(
                                            _fix_prompt, timeout_s=15, temperature=0.2, num_predict=500,
                                            system="你是文档编辑助手，只输出修正后的正文段落，禁止添加HTML注释或核查标记。")
                                        _fixed_body = re.sub(r"<!--.*?-->", "", str(_fixed_body or ""), flags=re.DOTALL).strip()
                                        if _fixed_body and len(_fixed_body) > 20:
                                            _lines = annotated.splitlines()
                                            _new_lines = []
                                            _skip = False
                                            _found = False
                                            for _cl in _lines:
                                                if _cl.strip().startswith(f"## {_sec_name}"):
                                                    _new_lines.append(_cl)
                                                    _new_lines.append("")
                                                    _new_lines.append(str(_fixed_body).strip())
                                                    _skip = True
                                                    _found = True
                                                    continue
                                                if _skip:
                                                    if _cl.strip().startswith("## "):
                                                        _new_lines.append("")
                                                        _new_lines.append(_cl)
                                                        _skip = False
                                                    continue
                                                _new_lines.append(_cl)
                                            if _found:
                                                annotated = "\n".join(_new_lines)
                                                _did_fix = True
                                                _fixed_count += 1
                                                logger.info(
                                                    "faithfulness_fixed section=%s claims=%d",
                                                    _sec_name, len(_claims))
                    except Exception as _e:
                        logger.warning("faithfulness_refix_failed section=%s err=%s",
                                       _sec_name, str(_e)[:100])

                if not _did_fix:
                    _unfixed.append(_w)
                    _marker = f"> ⚠️ **事实核查提醒[{_w['severity']}]**：{_reason}\n> 💡 {_w['remedy']}\n> 已尝试增量检索修正，仍未找到充分依据。请人工核实。\n\n"
                    _esc_sec = re.escape(f"## {_sec_name}")
                    _repl = f"## {_sec_name}\n{_marker}"
                    _new = re.sub(_esc_sec, _repl, annotated, count=1)
                    if _new != annotated:
                        annotated = _new

            if annotated != content:
                annotated = re.sub(r"<!--.*?-->", "", annotated, flags=re.DOTALL)
                try:
                    from pathlib import Path
                    (Path(output_dir) / "content.md").write_text(annotated, encoding="utf-8")
                except Exception as e:
                    logger.debug("faithfulness annotated write failed: %s", e)
                content = annotated
                if _fixed_count:
                    logger.info("faithfulness_summary fixed=%d unfixed=%d",
                                _fixed_count, len(_unfixed))

            return StepResult(
                success=True, content=content,
                data={
                    "checked": True,
                    "warnings": _warnings,
                    "fixed_count": _fixed_count,
                    "unfixed": _unfixed,
                },
            )

        except Exception as _e:
            logger.warning("faithfulness_check_failed err=%s", str(_e)[:200])
            return StepResult(success=False, error=str(_e), content=ctx.content)
