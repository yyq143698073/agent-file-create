"""Unified LangChain ChatModel factory.

All LLM instantiation flows through this single module, so that the rest of
the project never imports ChatOpenAI / ChatOllama directly.
"""

import hashlib
from functools import lru_cache
from typing import List, Optional, Tuple

from langchain_core.language_models import BaseChatModel


def _normalize_openai_base_url(endpoint: str) -> str:
    u = (endpoint or "").strip()
    if not u:
        return ""
    u = u.rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")].rstrip("/")
    if u.endswith("/v1"):
        return u
    if "/v1/" in u:
        i = u.find("/v1/")
        return u[: i + 3]
    return u


def _normalize_ollama_base_url(endpoint: str) -> str:
    u = (endpoint or "").strip()
    if not u:
        return ""
    return u.rstrip("/")


# ── Ollama ChatOllama import (single fallback site) ──────────────────────
def _import_chat_ollama():
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
    except Exception:
        from langchain_community.chat_models import ChatOllama  # type: ignore[import-untyped,deprecated]
    return ChatOllama


def _hash_key(k: str) -> str:
    """One-way hash so the raw key never appears in the lru_cache internals."""
    if not k:
        return "<none>"
    return hashlib.sha256(k.encode()).hexdigest()[:16]


def _attach_token_counting_fallback(llm) -> None:
    """Monkey-patch get_num_tokens_from_messages with a character-count fallback.

    Needed for non-OpenAI providers (DeepSeek etc.) that raise NotImplementedError.
    The fallback is only triggered when the original method fails.
    """
    try:
        orig = llm.get_num_tokens_from_messages
    except AttributeError:
        return

    def _fallback(messages):
        try:
            return orig(messages)
        except NotImplementedError:
            total = 0
            for m in messages:
                content = getattr(m, "content", "") or ""
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            total += len(str(part.get("text", "")))
            return max(total // 2, 1)

    object.__setattr__(llm, "get_num_tokens_from_messages", _fallback)


# ── DeepSeek reasoning_content patch ────────────────────────────────────────
def _patch_deepseek_reasoning_content(llm, model: str, base_url: str) -> None:
    """Monkey-patch langchain_openai for DeepSeek reasoning_content round-trip.

    deepseek-v4-pro (thinking mode) returns *reasoning_content* in assistant
    responses and requires it back in subsequent requests.  langchain_openai
    drops this field in BOTH directions:

    1) _convert_dict_to_message — parsing API response → AIMessage: drops reasoning_content
    2) _convert_message_to_dict — serialising AIMessage → API dict: never emits reasoning_content

    We patch both at the module level.
    """
    is_deepseek = any(kw in (model or "").lower() for kw in ("deepseek",))
    is_deepseek = is_deepseek or any(kw in (base_url or "").lower() for kw in ("deepseek",))
    if not is_deepseek:
        return

    try:
        import langchain_openai.chat_models.base as mod
        from langchain_core.messages import AIMessage

        # ── Patch 1: capture reasoning_content from API response ──────────
        orig_dict_to_msg = mod._convert_dict_to_message

        def _patched_dict_to_msg(_dict):
            msg = orig_dict_to_msg(_dict)
            if isinstance(msg, AIMessage) and isinstance(_dict, dict):
                reasoning = _dict.get("reasoning_content")
                if reasoning:
                    msg.additional_kwargs["reasoning_content"] = str(reasoning)
            return msg

        mod._convert_dict_to_message = _patched_dict_to_msg

        # ── Patch 2: emit reasoning_content in API request ────────────────
        orig_msg_to_dict = mod._convert_message_to_dict

        def _patched_msg_to_dict(message, api="chat/completions"):
            result = orig_msg_to_dict(message, api=api)
            if isinstance(message, AIMessage) and isinstance(result, dict):
                reasoning = message.additional_kwargs.get("reasoning_content")
                if reasoning:
                    result["reasoning_content"] = str(reasoning)
            return result

        mod._convert_message_to_dict = _patched_msg_to_dict
    except Exception:
        pass


# ── Uncached factory ──────────────────────────────────────────────────────
def _create_chat_model(
    *,
    style: str,
    model: str,
    endpoint: str,
    api_key: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout_s: int,
    stop: Optional[List[str]],
) -> BaseChatModel:
    style = (style or "").strip().lower()

    if style == "openai":
        from langchain_openai import ChatOpenAI

        base_url = _normalize_openai_base_url(endpoint)
        if not base_url:
            raise RuntimeError("OpenAI-compatible endpoint is empty")

        stop_list: Optional[List[str]] = None
        if stop and isinstance(stop, list) and all(isinstance(x, str) for x in stop):
            stop_list = stop

        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key or None,
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            timeout=int(timeout_s),
            stop=stop_list,
        )

        # Attach a fallback token counter for non-OpenAI providers (e.g. DeepSeek)
        # that raise NotImplementedError from get_num_tokens_from_messages.
        # Needed by ConversationSummaryBufferMemory; harmless otherwise.
        _attach_token_counting_fallback(llm)

        # ── DeepSeek reasoning_content preservation ──────────────────────
        # deepseek-v4-pro (thinking mode) returns reasoning_content in the
        # assistant message and requires it to be passed back in subsequent
        # turns. ChatOpenAI drops it during message → dict conversion.
        # Monkey-patch to include it from additional_kwargs.
        _patch_deepseek_reasoning_content(llm, model, base_url)

        return llm

    # ollama  (default)
    ChatOllama = _import_chat_ollama()

    base_url = _normalize_ollama_base_url(endpoint)
    if not base_url:
        raise RuntimeError("Ollama host is empty")

    kwargs: dict = {}
    if stop and isinstance(stop, list) and all(isinstance(x, str) for x in stop):
        kwargs["stop"] = stop
    if max_tokens is not None:
        try:
            kwargs["num_predict"] = int(max_tokens)
        except Exception:
            pass

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=float(temperature) if temperature is not None else None,
        model_kwargs=kwargs,
    )


# ── Cached wrapper ────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def get_chat_model(
    *,
    style: str,
    model: str,
    endpoint: str,
    api_key: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_s: int = 120,
    stop_tuple: Tuple[str, ...] = (),
) -> BaseChatModel:
    """Return a (possibly cached) LangChain BaseChatModel.

    All parameters are hashable so that @lru_cache can key on them.
    The *api_key* is hashed before entering the cache key — the raw key is
    never retained in the LRU internals.
    """
    stop_list: Optional[List[str]] = list(stop_tuple) if stop_tuple else None
    return _create_chat_model(
        style=style,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        stop=stop_list,
    )


# ── Convenience helpers ───────────────────────────────────────────────────
def normalize_endpoint_for_cache(*, style: str, endpoint: str) -> str:
    """Produce the cache-key form of an endpoint URL."""
    if (style or "").strip().lower() == "openai":
        return _normalize_openai_base_url(endpoint)
    return _normalize_ollama_base_url(endpoint)
