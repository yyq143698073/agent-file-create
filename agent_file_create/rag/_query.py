"""Query processing mixin for KnowledgeBase — rewrite, classification, HyDE, filters.

Extracted from kb.py via mixin pattern. All methods reference self.* attributes
set by KnowledgeBase.__init__.
"""

import hashlib
import json as _json
import logging
import re

from langchain_core.output_parsers import StrOutputParser

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
)
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.prompts import (
    HYDE_PROMPT as _HYDE_PROMPT,
    METADATA_FILTER_PROMPT as _METADATA_FILTER_PROMPT,
    MULTI_QUERY_PROMPT as _MULTI_QUERY_PROMPT,
    QUERY_REWRITE_PROMPT as _QUERY_REWRITE_PROMPT,
    QUERY_ROUTE_PROMPT as _QUERY_ROUTE_PROMPT,
    STEPBACK_PROMPT as _STEPBACK_PROMPT,
)

from agent_file_create.rag._utils import (
    query_concreteness as _query_concreteness,
    query_has_numbers as _query_has_numbers,
    query_has_specialized_terms as _query_has_specialized_terms,
    query_has_technical_terms as _query_has_technical_terms,
)

logger = logging.getLogger(__name__)


class QueryMixin:
    """Query-layer methods — analysis, rewrite, classification, HyDE expansion."""

    def _analyze_query(self, q: str) -> dict:
        """Analyze query characteristics for adaptive recall tuning."""
        return {
            "has_numbers": _query_has_numbers(q),
            "has_tech_terms": _query_has_technical_terms(q),
            "concreteness": _query_concreteness(q),
            "length": len(q),
        }

    def _llm_quick(self, prompt_template, inputs: dict, *, max_tokens: int = 200) -> str:
        """Lightweight LLM call for query-layer operations."""
        try:
            chain = prompt_template | self._get_answer_llm_for_short(max_tokens) | StrOutputParser()
            return (chain.invoke(inputs) or "").strip()
        except Exception:
            return ""

    def _get_answer_llm_for_short(self, max_tokens: int = 200):
        """Short-timeout LLM for quick query operations."""
        return get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.0,
            max_tokens=int(max_tokens),
            timeout_s=30,
        )

    def rewrite_query(self, question: str) -> str:
        """Rewrite a casual/spoken query into a precise search query.

        Example: "那个报销怎么搞的" → "费用报销流程的具体步骤是什么？"
        """
        q = str(question or "").strip()
        if len(q) < 8:
            return q
        result = self._llm_quick(_QUERY_REWRITE_PROMPT, {"question": q}, max_tokens=150)
        return result if result and len(result) >= 4 else q

    def generate_query_variants(self, question: str, n: int = 3) -> list[str]:
        """Generate multiple phrasings of the same question for multi-query retrieval.

        Example: "研发投入占比" →
          ["2024年研发投入占总预算比例", "研发支出在财务预算中的比重", "R&D预算分配情况"]
        """
        q = str(question or "").strip()
        if len(q) < 6:
            return [q]
        result = self._llm_quick(_MULTI_QUERY_PROMPT, {"question": q, "n": str(n)}, max_tokens=250)
        if not result:
            return [q]
        variants: list[str] = []
        for line in result.splitlines():
            v = re.sub(r"^\d+[\.\)、\s]*", "", line).strip()
            if v and len(v) >= 4:
                variants.append(v)
        if not variants:
            return [q]
        seen: set[str] = set()
        uniq: list[str] = []
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                uniq.append(v)
        return uniq[:n]

    def generate_stepback_question(self, question: str) -> str:
        """Generate a higher-level background question for broader retrieval.

        Example: "2024年研发投入占比下降原因" →
          "公司研发投入的影响因素和决策依据有哪些？"
        """
        q = str(question or "").strip()
        if len(q) < 15:
            return q
        result = self._llm_quick(_STEPBACK_PROMPT, {"question": q}, max_tokens=150)
        return result if result and len(result) >= 6 else q

    def classify_query(self, question: str) -> str:
        """Classify query type for routing.

        Returns: conceptual | comparison | summary | multi_document | how_to | fact_lookup
        """
        q = str(question or "").strip()
        if len(q) < 5:
            return "fact_lookup"
        # Conceptual / definitional questions: "是什么", "应该是怎样的", "如何定义"
        if any(kw in q for kw in ["是什么", "什么是", "应该是", "如何定义", "定义",
                                   "概念", "框架", "体系", "结构", "组成",
                                   "特征", "特点", "特性", "怎样的", "怎么样"]):
            return "conceptual"
        if any(kw in q for kw in ["比较", "对比", "区别", "异同", "vs", "VS", "优缺点"]):
            return "comparison"
        if any(kw in q for kw in ["总结", "汇总", "概述", "概括", "归纳"]):
            return "summary"
        if any(kw in q for kw in ["怎么", "如何", "步骤", "流程", "方法", "操作"]):
            return "how_to"
        result = self._llm_quick(_QUERY_ROUTE_PROMPT, {"question": q}, max_tokens=30)
        r = (result or "").strip().lower()
        valid = {"conceptual", "fact_lookup", "comparison", "summary", "multi_document", "how_to"}
        return r if r in valid else "fact_lookup"

    def extract_metadata_filters(self, question: str) -> dict:
        """Extract implicit metadata filters from natural language.

        Example: "制度类文档中的风险管理政策" → {"doc_type": "制度"}
                 "2024年的财务报告" → {"time_range": "2024"}
        """
        q = str(question or "").strip()
        if len(q) < 6:
            return {}
        result = self._llm_quick(_METADATA_FILTER_PROMPT, {"question": q}, max_tokens=150)
        if not result or not result.startswith("{"):
            return {}
        try:
            obj = _json.loads(result)
            out: dict = {}
            if isinstance(obj, dict):
                if isinstance(obj.get("doc_type"), str) and obj["doc_type"].strip():
                    out["doc_type"] = obj["doc_type"].strip()
                if isinstance(obj.get("source"), str) and obj["source"].strip():
                    out["source"] = obj["source"].strip()
                if isinstance(obj.get("time_range"), str) and obj["time_range"].strip():
                    out["time_range"] = obj["time_range"].strip()
            return out
        except Exception:
            return {}

    def _hyde_expand(self, question: str) -> str:
        """Generate a hypothetical answer and return it as an expanded search query.

        HyDE (Hypothetical Document Embeddings) bridges the vocabulary gap
        between short queries and document chunks by first generating a
        plausible answer, then using that answer's embedding for retrieval.

        Results are cached by question MD5 hash to avoid redundant LLM calls.
        """
        q = str(question or "").strip()
        if len(q) < 10:
            return q
        key = hashlib.md5(q.encode("utf-8")).hexdigest()
        if key in self._hyde_cache:
            self._hyde_cache.move_to_end(key)
            return self._hyde_cache[key]
        try:
            chain = _HYDE_PROMPT | self._get_answer_llm() | StrOutputParser()
            hypothetical = (chain.invoke({"question": q}) or "").strip()
            if hypothetical and len(hypothetical) >= 15:
                result = hypothetical[:600]
            else:
                result = q
        except Exception:
            result = q
        self._hyde_cache[key] = result
        self._hyde_cache.move_to_end(key)
        while len(self._hyde_cache) > self._hyde_cache_MAX:
            self._hyde_cache.popitem(last=False)
        return result
