"""Answer generation mixin for KnowledgeBase — context assembly, compression, answer*.

Extracted from kb.py via mixin pattern. All methods reference self.* attributes
set by KnowledgeBase.__init__.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
)
from agent_file_create.llm_client import call_llm
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.prompts import (
    ANSWER_COT_PROMPT as _ANSWER_COT_PROMPT,
    ANSWER_PROMPT as _ANSWER_PROMPT,
    Answer,
    Citation,
    DECOMPOSE_PROMPT as _DECOMPOSE_PROMPT,
    HYDE_PROMPT as _HYDE_PROMPT,
)
from agent_file_create.rag.reranker import rerank
from agent_file_create.rag.store import Hit

from agent_file_create.rag._utils import normalize_kb as _normalize_kb

logger = logging.getLogger(__name__)


class AnswerMixin:
    """Answer generation methods — context assembly, reply formatting, CoT reasoning."""

    def _assemble_context(
        self, hits: list[Hit], base_k: int, max_context_chars: int = 5200
    ) -> tuple[str, list[Citation]]:
        """Shared context assembly: dedup by doc → merge adjacent → sort → truncate.

        Used by answer(), answer_with_reasoning(), and decompose_and_answer().
        """
        citations: list[Citation] = []
        picked: list[Hit] = []
        per_doc: dict[str, int] = {}
        overflow: list[Hit] = []
        for h in hits:
            did = str(h.doc_id or "")
            c = int(per_doc.get(did, 0))
            if c < 2:
                per_doc[did] = c + 1
                picked.append(h)
            else:
                overflow.append(h)
            if len(picked) >= base_k:
                break
        if len(picked) < base_k:
            for h in overflow:
                picked.append(h)
                if len(picked) >= base_k:
                    break

        by_doc: dict[str, list[Hit]] = {}
        for h in picked:
            by_doc.setdefault(str(h.doc_id or ""), []).append(h)
        segments: list[list[Hit]] = []
        for did, hs in by_doc.items():
            hs.sort(key=lambda x: int(x.chunk_index or 0))
            cur: list[Hit] = []
            last_i: Optional[int] = None
            for h in hs:
                ci = int(h.chunk_index or 0)
                if cur and (last_i is not None) and (ci - last_i <= 1):
                    cur.append(h)
                    last_i = ci
                    continue
                if cur:
                    segments.append(cur)
                cur = [h]
                last_i = ci
            if cur:
                segments.append(cur)
        segments.sort(key=lambda g: max(float(x.score) for x in g), reverse=True)

        blocks: list[str] = []
        used = 0
        idx2 = 1
        for group in segments:
            if not group:
                continue
            h0 = group[0]
            meta0 = h0.meta if isinstance(h0.meta, dict) else {}
            sec0 = str(h0.section_path or "").strip() or str((meta0 or {}).get("title") or "").strip() or "-"
            did = str(h0.doc_id or "").strip() or "-"
            parts: list[str] = []
            for h in group:
                snip = (h.content or "").strip()
                if len(snip) > 900:
                    snip = snip[:900] + "…"
                meta = h.meta if isinstance(h.meta, dict) else {}
                sec = str(h.section_path or "").strip() or str((meta or {}).get("title") or "").strip() or sec0
                citations.append(Citation(
                    doc_id=h.doc_id, chunk_id=h.chunk_id, section_path=sec,
                    score=float(h.score), snippet=snip,
                    doc_name=str(meta.get("title") or h.doc_id or "")[:60],
                ))
                parts.append(snip)
            body = "\n\n".join([p for p in parts if p]).strip()
            score = max(float(x.score) for x in group)
            head = f"[{idx2}] doc={did} section={sec0} score={score:.3f}"
            block = (head + "\n" + body).strip() if body else head
            if used + len(block) + 2 > int(max_context_chars or 0):
                break
            blocks.append(block)
            used += len(block) + 2
            idx2 += 1

        return "\n\n".join(blocks).strip(), citations

    # ── Context Compression (CRAG-style) ──────────────────────────────────────

    def compress_context(self, *, kb: str, query: str, top_k: int = 15,
                          max_chars: int = 1500) -> str:
        """Search → decompose chunks into sentences → filter irrelevant → recompose.

        Reduces noise from retrieved chunks before passing to the LLM, improving
        answer quality and reducing hallucinations from irrelevant content.
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return ""

        # Step 1: Retrieve a larger candidate pool
        hits = self.search_adaptive(kb=kb2, query=q, top_k=top_k)
        if not hits:
            hits = self.search(kb=kb2, query=q, top_k=top_k)
        if not hits:
            return ""

        # Step 2: Decompose — collect all sentences from hits
        all_sentences: list[str] = []
        for h in hits:
            content = str(h.content or "").strip()
            if not content:
                continue
            for sent in re.split(r"[。！？.!?\n]+", content):
                sent = sent.strip()
                if len(sent) >= 8:
                    all_sentences.append(sent)

        if not all_sentences:
            return "\n\n".join(str(h.content or "") for h in hits[:3])

        # Step 3: LLM picks relevant sentences (1 LLM call)
        sentences_text = "\n".join(
            f"[S{i+1}] {s}" for i, s in enumerate(all_sentences[:30]))
        prompt = (
            "从以下检索到的句子中筛选出与问题相关的句子，过滤无关内容。\n\n"
            f"问题：{q[:300]}\n\n候选句子：\n{sentences_text[:3000]}\n\n"
            "输出相关句子序号(逗号分隔,如S1,S3,S5)。无相关回复NONE。"
        )
        raw = call_llm(prompt, timeout_s=15, temperature=0.0, num_predict=100,
                       system="你是一个中文文档处理助手。只输出相关句子序号。")
        indices: set[int] = set()
        for m in re.findall(r"S?(\d+)", raw or ""):
            idx = int(m) - 1
            if 0 <= idx < len(all_sentences):
                indices.add(idx)

        if not indices:
            return "\n\n".join(str(h.content or "") for h in hits[:3])

        # Step 4: Recompose — join relevant sentences
        refined = ""
        for i in sorted(indices):
            s = all_sentences[i]
            if len(refined) + len(s) + 2 > max_chars:
                break
            refined += s + "。"
        return refined.strip()

    def answer_smart(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 6,
        max_context_chars: int = 5200,
        filters: Optional[dict] = None,
    ) -> Answer:
        """Intelligent query routing: classify → rewrite → fetch → assemble → answer.

        Routes to the best retrieval strategy based on query type:
        - fact_lookup: rewritten query + direct search (fast, precise)
        - comparison: multi-query → merge results from both sides
        - summary: step-back search → broader context
        - how_to: rewritten query + metadata-filtered search
        - multi_document: multi-query + step-back combined
        """
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        base_k = max(3, int(top_k or 0))

        # Step 1: Extract metadata filters from natural language
        nl_filters = self.extract_metadata_filters(q)
        merged_filters: dict = dict(filters or {})
        merged_filters.update(nl_filters)

        # Step 2: Classify query type
        qtype = self.classify_query(q)

        # Step 3: Route to retrieval strategy
        if qtype == "conceptual":
            # Conceptual questions need broader context, not just keyword match.
            # Combine multi-query + stepback + HyDE to cast the widest net.
            mq_hits = self.search_multi_query(kb=kb2, question=q, top_k=base_k * 2, n_variants=3, filters=merged_filters)
            sb_hits = self.search_with_stepback(kb=kb2, question=q, top_k=base_k, filters=merged_filters)
            try:
                hyde_q = self._hyde_expand(q)
                hyde_hits = self.search(kb=kb2, query=hyde_q, top_k=base_k * 2, filters=merged_filters)
            except Exception:
                hyde_hits = []
            all_hits: dict[str, Hit] = {}
            for h in mq_hits + sb_hits + hyde_hits:
                if h.chunk_id not in all_hits or h.score > all_hits[h.chunk_id].score:
                    all_hits[h.chunk_id] = h
            hits = sorted(all_hits.values(), key=lambda x: x.score, reverse=True)[:max(1, base_k * 4)]
        elif qtype == "comparison":
            hits = self.search_multi_query(kb=kb2, question=q, top_k=base_k, n_variants=4, filters=merged_filters)
        elif qtype == "summary":
            hits = self.search_with_stepback(kb=kb2, question=q, top_k=base_k, filters=merged_filters)
        elif qtype == "multi_document":
            mq_hits = self.search_multi_query(kb=kb2, question=q, top_k=base_k * 2, n_variants=3, filters=merged_filters)
            sb_hits = self.search_with_stepback(kb=kb2, question=q, top_k=base_k, filters=merged_filters)
            merged: dict[str, Hit] = {}
            for h in mq_hits + sb_hits:
                if h.chunk_id not in merged or h.score > merged[h.chunk_id].score:
                    merged[h.chunk_id] = h
            hits = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:max(1, base_k * 3)]
        elif qtype == "how_to":
            rewritten = self.rewrite_query(q)
            hits = self.search(kb=kb2, query=rewritten, top_k=max(10, base_k * 3), filters=merged_filters)
        else:  # fact_lookup
            rewritten = self.rewrite_query(q)
            hits = self.search(kb=kb2, query=rewritten, top_k=max(10, base_k * 3), filters=merged_filters)

        # Step 4: Rerank → assemble → generate
        hits = rerank(q, hits, top_k=max(10, base_k * 3))
        ctx, citations = self._assemble_context(hits, base_k, max_context_chars)

        # ── Low-score degradation: if reranker-normalized scores are still poor ──
        # Note: rerank() applies _score_norm() which normalizes to [0,1].
        # Raw RRF scores are ~0.01-0.03 (ceiling 0.03 with k=60), which would
        # always trigger a low-score warning if we didn't wait for normalization.
        scores = [float(c.score or 0) for c in citations if c.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        temp = 0.2

        # Only trigger degradation if ALL scores are sub-0.05 AFTER reranker normalization.
        # A single chunk > 0.05 means the retrieval found something genuinely relevant.
        max_score = max(scores) if scores else 0
        has_good_hit = max_score > 0.05
        if (avg_score < 0.05 or not ctx) and not has_good_hit:
            # Retrieval quality is very low — allow LLM to use more own knowledge
            low_score_note = (
                "\n\n⚠️ 检索质量提示：以下片段与问题的相关度很低（平均分 {:.3f}）。"
                "请优先使用你自己的知识给出框架性回答，检索片段仅作微弱参考。"
                "在回答开头标注「以下回答基于通用知识，知识库中未找到高相关度内容」。"
            ).format(avg_score) if scores else ""
            ctx = (ctx or "（未命中）") + low_score_note
            temp = 0.5  # Higher temperature for more creative synthesis
            logger.info("answer_smart low_score_degrade kb=%s avg_score=%.4f", kb2, avg_score)

        if not ctx:
            return Answer(kb=kb2, question=q, answer="未找到相关信息。建议你更换关键词、缩小范围（doc_type/doc_id），或先把相关文档上传入库。", citations=[])

        text = (
            _ANSWER_PROMPT
            | get_chat_model(
                style=CONTENT_API_STYLE, model=CONTENT_MODEL_NAME,
                endpoint=CONTENT_API_ENDPOINT, api_key=CONTENT_API_KEY,
                temperature=temp, max_tokens=520, timeout_s=120,
            )
            | StrOutputParser()
        ).invoke({"context": ctx or "（未命中）", "question": q or "（空）", "kb": kb2})
        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()
        if not out:
            out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"

        uniq: list[Citation] = []
        seen = set()
        for c in citations:
            k = str(c.chunk_id or "")
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(c)
            if len(uniq) >= 6:
                break
        return Answer(kb=kb2, question=q, answer=out, citations=uniq)

    def answer(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 6,
        max_context_chars: int = 5200,
        filters: Optional[dict] = None,
    ) -> Answer:
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        base_k = max(3, int(top_k or 0))
        logger.info("kb_answer_start kb=%s question=%.80s top_k=%d", kb2, q, top_k)
        hits = self.search_with_context(kb=kb2, query=q, top_k=max(20, base_k * 5), context_window=2, filters=filters)
        logger.info("kb_answer_search_done kb=%s hits=%d", kb2, len(hits))
        hits = rerank(q, hits, top_k=max(12, base_k * 3))
        ctx, citations = self._assemble_context(hits, base_k, max_context_chars)
        logger.info("kb_answer_context kb=%s hits_after_rerank=%d context_chars=%d citations=%d", kb2, len(hits), len(ctx) if ctx else 0, len(citations))
        if not ctx:
            logger.warning("kb_answer_no_context kb=%s question=%.80s hits_before=%d", kb2, q, len(hits))
            return Answer(kb=kb2, question=q, answer="未找到相关信息。建议你更换关键词、缩小范围（doc_type/doc_id），或先把相关文档上传入库。", citations=[])

        logger.info("kb_answer_llm_start kb=%s context_chars=%d", kb2, len(ctx) if ctx else 0)
        try:
            text = (
                _ANSWER_PROMPT
                | get_chat_model(
                    style=CONTENT_API_STYLE,
                    model=CONTENT_MODEL_NAME,
                    endpoint=CONTENT_API_ENDPOINT,
                    api_key=CONTENT_API_KEY,
                    temperature=0.2,
                    max_tokens=420,
                    timeout_s=120,
                )
                | StrOutputParser()
            ).invoke({"context": ctx or "（未命中）", "question": q or "（空）", "kb": kb2})
        except Exception as e:
            logger.error("kb_answer_llm_failed kb=%s err=%s", kb2, str(e)[:200])
            return Answer(kb=kb2, question=q, answer=f"答案生成失败：{str(e)[:120]}", citations=citations)
        logger.info("kb_answer_done kb=%s answer_chars=%d", kb2, len(text or ""))
        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()
        if not out:
            out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"
        elif out.startswith("{"):
            low = out.lower()
            if ("不确定" not in out) and ("未命中" not in out) and ("unknown" not in low):
                if len(ctx) >= 300:
                    out = "模型未返回可解析回答。已命中相关片段，请尝试缩小问题范围或指定文档后重试。"
                else:
                    out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"

        uniq: list[Citation] = []
        seen = set()
        for c in citations:
            k = str(c.chunk_id or "")
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(c)
            if len(uniq) >= 6:
                break
        return Answer(kb=kb2, question=q, answer=out, citations=uniq)

    def _get_answer_llm(self):
        """Cached LLM instance for answer generation."""
        return get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.2,
            max_tokens=420,
            timeout_s=120,
        )

    def answer_with_reasoning(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 6,
        max_context_chars: int = 5200,
        use_hyde: bool = True,
        filters: Optional[dict] = None,
    ) -> Answer:
        """Answer with chain-of-thought reasoning and optional HyDE retrieval.

        Compared to answer(), this method:
        - Uses HyDE to expand the query before retrieval (if use_hyde=True)
        - Requires the LLM to show its reasoning steps before the final answer
        - Self-verifies each claim against retrieved evidence
        """
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        base_k = max(3, int(top_k or 0))

        # HyDE: expand query with hypothetical answer for better recall
        search_query = self._hyde_expand(q) if use_hyde else q

        hits = self.search(kb=kb2, query=search_query, top_k=max(10, base_k * 3), filters=filters)
        hits = rerank(q, hits, top_k=max(10, base_k * 3))
        ctx, citations = self._assemble_context(hits, base_k, max_context_chars)
        if not ctx:
            return Answer(kb=kb2, question=q, answer="未找到相关信息。建议你更换关键词、缩小范围（doc_type/doc_id），或先把相关文档上传入库。", citations=[])

        text = (
            _ANSWER_COT_PROMPT
            | get_chat_model(
                style=CONTENT_API_STYLE,
                model=CONTENT_MODEL_NAME,
                endpoint=CONTENT_API_ENDPOINT,
                api_key=CONTENT_API_KEY,
                temperature=0.1,
                max_tokens=900,
                timeout_s=120,
            )
            | StrOutputParser()
        ).invoke({"context": ctx or "（未命中）", "question": q or "（空）", "kb": kb2})
        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()
        if not out:
            out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"

        uniq: list[Citation] = []
        seen = set()
        for c in citations:
            k = str(c.chunk_id or "")
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(c)
            if len(uniq) >= 6:
                break
        return Answer(kb=kb2, question=q, answer=out, citations=uniq)

    def _decompose_question(self, question: str) -> list[str]:
        """Decompose a complex question into 2-4 simpler sub-questions."""
        q = str(question or "").strip()
        if len(q) < 20:
            return [q]
        try:
            chain = _DECOMPOSE_PROMPT | self._get_answer_llm() | StrOutputParser()
            result = (chain.invoke({"question": q}) or "").strip()
        except Exception:
            return [q]
        if not result or result.upper().startswith("SIMPLE"):
            return [q]
        subs: list[str] = []
        for line in result.splitlines():
            sub = re.sub(r"^\d+[\.\)、\s]*", "", line).strip()
            if sub and len(sub) >= 5:
                subs.append(sub)
        return subs if subs else [q]

    def decompose_and_answer(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 4,
        use_hyde: bool = True,
        filters: Optional[dict] = None,
    ) -> Answer:
        """For complex questions: decompose → retrieve per sub-Q → synthesize.

        Best for comparison, multi-aspect analysis, or cause-effect questions.
        """
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        subs = self._decompose_question(q)
        if len(subs) <= 1:
            return self.answer_with_reasoning(kb=kb, question=q, top_k=top_k, use_hyde=use_hyde, filters=filters)

        # Retrieve for each sub-question in parallel
        sub_results: list[dict] = []
        max_w = min(4, len(subs))
        if max_w > 1:
            with ThreadPoolExecutor(max_workers=max_w) as ex:
                futures = {
                    ex.submit(self.answer, kb=kb2, question=sub, top_k=max(3, top_k), max_context_chars=2400, filters=filters): sub
                    for sub in subs
                }
                for future in as_completed(futures):
                    try:
                        sub_ans = future.result()
                    except Exception:
                        sub_ans = Answer(kb=kb2, question=futures[future], answer="子问题检索失败", citations=[])
                    sub_results.append({"question": futures[future], "answer": sub_ans.answer, "citations": sub_ans.citations})
        else:
            for sub in subs:
                sub_ans = self.answer(kb=kb2, question=sub, top_k=max(3, top_k), max_context_chars=2400, filters=filters)
                sub_results.append({"question": sub, "answer": sub_ans.answer, "citations": sub_ans.citations})

        # Synthesize
        parts = []
        for i, sr in enumerate(sub_results):
            parts.append(f"子问题{i+1}：{sr['question']}\n初步回答：{sr['answer']}")
        synthesis_context = "\n\n".join(parts)

        synth_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个中文文档处理助手。擅长综合多角度信息。"),
            ("human", """\
基于以下子问题的分析结果，综合回答原始问题。要求：
1) 融合各子问题的关键发现，给出连贯的整体回答
2) 标注不同观点或证据之间的关联（因果关系、对比、互补等）
3) 如有矛盾，指出并给出最可能的结论
4) 末尾追加一行：依据：<引用来源（最多3条）>

原始问题：{question}

子问题分析：
{synthesis_context}

综合回答："""),
        ])
        try:
            text = (
                synth_prompt
                | get_chat_model(
                    style=CONTENT_API_STYLE,
                    model=CONTENT_MODEL_NAME,
                    endpoint=CONTENT_API_ENDPOINT,
                    api_key=CONTENT_API_KEY,
                    temperature=0.2,
                    max_tokens=700,
                    timeout_s=120,
                )
                | StrOutputParser()
            ).invoke({"question": q, "synthesis_context": synthesis_context})
        except Exception:
            text = "\n\n".join([f"**{sr['question']}**\n{sr['answer']}" for sr in sub_results])

        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()

        all_citations: list[Citation] = []
        seen = set()
        for sr in sub_results:
            for c in (sr.get("citations") or []):
                k = str(c.chunk_id or "")
                if not k or k in seen:
                    continue
                seen.add(k)
                all_citations.append(c)
                if len(all_citations) >= 6:
                    break
            if len(all_citations) >= 6:
                break

        return Answer(kb=kb2, question=q, answer=out or "综合回答生成失败，请尝试更具体的问题。", citations=all_citations)
