import hashlib
import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from agent_file_create.config import (
    API_GENERATE_ENDPOINT,
    MODEL_NAME,
    OLLAMA_HOST,
    OPENAI_API_ENDPOINT,
    OPENAI_MODEL_NAME,
    OPENAI_API_KEY,
    MODEL_TIMEOUT,
)

logger = logging.getLogger(__name__)

from agent_file_create.prompts import SYSTEM_ASSISTANT as DEFAULT_SYSTEM_PROMPT


def _extract_final_answer(thinking: str) -> str:
    t = (thinking or "").strip()
    if not t:
        return ""
    markers = [
        "Final Answer:",
        "Final answer:",
        "FINAL ANSWER:",
        "最终答案：",
        "答案：",
        "回答：",
        "结论：",
    ]
    best_i = -1
    best_m = ""
    for m in markers:
        i = t.rfind(m)
        if i > best_i:
            best_i = i
            best_m = m
    if best_i >= 0:
        return t[best_i + len(best_m) :].strip()
    return ""


def _env_flag(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _normalize_stop_list(stop: Optional[List[str]]) -> Tuple[str, ...]:
    if not stop or not isinstance(stop, list):
        return tuple()
    out: list[str] = []
    for s in stop:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if t:
            out.append(t)
    return tuple(out)


def _resolve_llm_config(
    *,
    api_style: Optional[str],
    api_endpoint: Optional[str],
    model_name: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, str, str]:
    style = (api_style or "").strip().lower()
    endpoint_in = (api_endpoint or "").strip()

    if not style:
        if endpoint_in and "/v1" in endpoint_in:
            style = "openai"
        elif (OPENAI_API_ENDPOINT or "").strip():
            style = "openai"
        else:
            style = "ollama"

    if style == "openai":
        endpoint = endpoint_in or (OPENAI_API_ENDPOINT or "").strip()
        if not endpoint:
            base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
            if base_url:
                endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        model = (model_name or (OPENAI_MODEL_NAME or "").strip() or MODEL_NAME).strip()
        key = (api_key or (OPENAI_API_KEY or "").strip()).strip()
        return "openai", endpoint, model, key

    model = (model_name or MODEL_NAME).strip()
    endpoint = (endpoint_in or API_GENERATE_ENDPOINT).strip()
    return "ollama", endpoint, model, ""


def _system_to_inject(system: Optional[str]) -> Optional[str]:
    if system is None:
        return DEFAULT_SYSTEM_PROMPT
    if isinstance(system, str) and system == "":
        return None
    return str(system)


def _has_system_message(msgs_in: list[Any]) -> bool:
    from langchain_core.messages import SystemMessage, BaseMessage

    for m in msgs_in:
        if isinstance(m, BaseMessage):
            if isinstance(m, SystemMessage):
                return True
            continue
        if isinstance(m, dict):
            role = str(m.get("role") or "").strip().lower()
            if role == "system":
                return True
    return False


def _get_msg_images(m: dict) -> list[str]:
    v = m.get("images_base64", None)
    if v is None:
        v = m.get("images", None)
    if v is None:
        return []
    if isinstance(v, str):
        t = v.strip()
        return [t] if t else []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if not isinstance(x, str):
                continue
            t = x.strip()
            if t:
                out.append(t)
        return out
    return []


def _find_last_user_dict_index(msgs_in: list[Any]) -> int:
    last_i = -1
    for i, m in enumerate(msgs_in):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        if role in {"user", ""}:
            last_i = i
    return last_i




def _build_messages(prompt: str, *, system: Optional[str], images_base64: Optional[List[str]], style: str):
    from langchain_core.messages import HumanMessage, SystemMessage

    sys_prompt = _system_to_inject(system)
    sys = SystemMessage(content=sys_prompt) if sys_prompt is not None else None

    if images_base64:
        if style == "openai":
            parts = [{"type": "text", "text": prompt or ""}]
            for b64 in images_base64:
                b = str(b64 or "").strip()
                if not b:
                    continue
                parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}})
            if sys is None:
                return [HumanMessage(content=parts)]
            return [sys, HumanMessage(content=parts)]

        if sys is None:
            return [HumanMessage(content=prompt or "", additional_kwargs={"images": images_base64})]
        return [sys, HumanMessage(content=prompt or "", additional_kwargs={"images": images_base64})]

    if sys is None:
        return [HumanMessage(content=prompt or "")]
    return [sys, HumanMessage(content=prompt or "")]


def _coerce_messages(messages: Any, *, system: Optional[str], style: str, images_base64: Optional[List[str]]):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

    msgs_in: list[Any] = []
    if messages is None:
        msgs_in = []
    elif isinstance(messages, list):
        msgs_in = messages
    else:
        raise TypeError("messages 必须是 list")

    out: list[BaseMessage] = []
    has_system = _has_system_message(msgs_in)
    sys_prompt = _system_to_inject(system) if not has_system else None
    if sys_prompt is not None:
        out.append(SystemMessage(content=sys_prompt))

    global_images: list[str] = []
    if images_base64 and isinstance(images_base64, list):
        for x in images_base64:
            if not isinstance(x, str):
                continue
            t = x.strip()
            if t:
                global_images.append(t)
    last_user_dict_i = _find_last_user_dict_index(msgs_in) if global_images else -1

    for i, m in enumerate(msgs_in):
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if isinstance(m, dict):
            role = str(m.get("role") or "").strip().lower()
            content = m.get("content")
            if role == "system":
                out.append(SystemMessage(content=str(content or "")))
            elif role == "assistant":
                out.append(AIMessage(content=str(content or "")))
            else:
                msg_images = _get_msg_images(m)
                if not msg_images and global_images and i == last_user_dict_i:
                    msg_images = global_images

                if msg_images and role in {"user", ""}:
                    if style == "openai":
                        if isinstance(content, list):
                            parts = list(content)
                        else:
                            parts = [{"type": "text", "text": str(content or "")}]
                        for b64 in msg_images:
                            parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}})
                        out.append(HumanMessage(content=parts))
                    else:
                        out.append(HumanMessage(content=str(content or ""), additional_kwargs={"images": msg_images}))
                else:
                    out.append(
                        HumanMessage(
                            content=content
                            if (style == "openai" and isinstance(content, list))
                            else str(content or "")
                        )
                    )
            continue
        raise TypeError("messages 元素必须是 dict 或 LangChain Message")

    if not out:
        out = _build_messages("", system=system, images_base64=images_base64, style=style)
    return out


class LLMCallError(RuntimeError):
    pass


def _normalize_endpoint_for_cache(*, style: str, endpoint: str) -> str:
    return normalize_endpoint_for_cache(style=style, endpoint=endpoint) or (OLLAMA_HOST or "").strip()


def _get_chat_model_cached(
    *,
    style: str,
    model: str,
    endpoint_norm: str,
    api_key: str,
    temperature: Optional[float],
    num_predict: Optional[int],
    timeout_s: int,
    stop_tuple: Tuple[str, ...] = (),
):
    return get_chat_model(
        style=style,
        model=model,
        endpoint=endpoint_norm,
        api_key=api_key,
        temperature=temperature,
        max_tokens=num_predict,
        timeout_s=timeout_s,
        stop_tuple=stop_tuple,
    )


def call_llm(
    prompt: str,
    *,
    messages: Any = None,
    images_base64: Optional[List[str]] = None,
    timeout_s: Optional[int] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    stop: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    api_style: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    r = call_llm_ex(
        prompt,
        messages=messages,
        images_base64=images_base64,
        timeout_s=timeout_s,
        system=system,
        temperature=temperature,
        num_predict=num_predict,
        model_name=model_name,
        api_endpoint=api_endpoint,
        api_style=api_style,
        api_key=api_key,
    )
    if isinstance(r, dict) and r.get("success"):
        return str(r.get("text") or "")
    err = ""
    if isinstance(r, dict):
        err = str(r.get("error") or "").strip()
    return json.dumps({"error": err or "unknown_error"}, ensure_ascii=False)






def call_llm_ex(
    prompt: str,
    *,
    messages: Any = None,
    images_base64: Optional[List[str]] = None,
    timeout_s: Optional[int] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    model_name: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    api_style: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    t0 = time.perf_counter()
    timeout_val = int(timeout_s or MODEL_TIMEOUT)
    style, endpoint, model, key = _resolve_llm_config(
        api_style=api_style,
        api_endpoint=api_endpoint,
        model_name=model_name,
        api_key=api_key,
    )
    endpoint_norm = _normalize_endpoint_for_cache(style=style, endpoint=endpoint)

    last_err: str = ""
    for attempt in range(2):
        try:
            llm = _get_chat_model_cached(
                style=style,
                model=model,
                endpoint_norm=endpoint_norm,
                api_key=key,
                temperature=temperature,
                num_predict=num_predict,
                timeout_s=timeout_val,
            )
            if messages is not None:
                msgs = _coerce_messages(messages, system=system, style=style, images_base64=images_base64)
            else:
                msgs = _build_messages(prompt or "", system=system, images_base64=images_base64, style=style)

            resp = llm.invoke(msgs)
            text = getattr(resp, "content", "")
            if not isinstance(text, str) or not text.strip():
                raise LLMCallError("模型返回空 content")
            t1 = time.perf_counter()
            if t1 - t0 >= 8:
                logger.info(f"llm_call_slow seconds={t1 - t0:.2f} prompt_chars={len(prompt or '')} style={style} model={model}")
            return {"success": True, "text": text, "meta": {"style": style, "model": model, "seconds": t1 - t0}}
        except Exception as e:
            last_err = str(e)[:240]
            # Fallback: if Ollama fails and OpenAI endpoint is configured, switch
            if style == "ollama" and (attempt >= 0):
                oai_endpoint = (OPENAI_API_ENDPOINT or "").strip()
                if oai_endpoint and oai_endpoint != (API_GENERATE_ENDPOINT or "").strip():
                    logger.info(f"llm_ollama_fallback switching to openai after {attempt + 1} attempts")
                    style = "openai"
                    endpoint = oai_endpoint
                    endpoint_norm = _normalize_endpoint_for_cache(style="openai", endpoint=endpoint)

                    # Map Ollama model names to DeepSeek-compatible models
                    fallback_model = (model_name or model or "").strip()
                    fallback_low = fallback_model.lower()
                    if any(v in fallback_low for v in ("minicpm", "llava", "bakllava", "vision", "visual")):
                        # Vision model — DeepSeek is text-only; strip images and use text model
                        model = "deepseek-v4-flash"
                        images_base64 = None
                    elif any(v in fallback_low for v in ("qwen", "llama", "mistral", "gemma", "phi", "yi", "glm", "deepseek-r1", "deepseek-v3")):
                        model = "deepseek-v4-flash"
                    else:
                        model = "deepseek-v4-flash"

                    key = (api_key or OPENAI_API_KEY or "").strip()
                    # Don't count this as a failed attempt
                    attempt -= 1
                    time.sleep(0.5)
                    continue
            if attempt < 2:
                time.sleep(0.8 + 0.8 * attempt)
                continue
            t1 = time.perf_counter()
            try:
                logger.warning(f"llm_call_failed err={last_err}")
            except Exception:
                pass
            return {"success": False, "text": "", "error": last_err, "meta": {"style": style, "model": model, "seconds": t1 - t0}}
    t1 = time.perf_counter()
    return {"success": False, "text": "", "error": last_err or "unknown_error", "meta": {"style": style, "model": model, "seconds": t1 - t0}}


# ── ChatModel factory (merged from llm_factory.py) ──────────────────────────

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

        _attach_token_counting_fallback(llm)
        _patch_deepseek_reasoning_content(llm, model, base_url)
        return llm

    # ollama (default)
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


def normalize_endpoint_for_cache(*, style: str, endpoint: str) -> str:
    """Produce the cache-key form of an endpoint URL."""
    if (style or "").strip().lower() == "openai":
        return _normalize_openai_base_url(endpoint)
    return _normalize_ollama_base_url(endpoint)
