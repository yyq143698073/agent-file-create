import json
import logging
import os
import time
from typing import Any, List, Optional, Tuple

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

DEFAULT_SYSTEM_PROMPT = "你是一个中文助手，只输出中文。"


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
    u = u.rstrip("/")
    if u.endswith("/api/generate"):
        return u[: -len("/api/generate")].rstrip("/")
    if u.endswith("/api/chat"):
        return u[: -len("/api/chat")].rstrip("/")
    return u


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
    from .llm_factory import normalize_endpoint_for_cache

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
    from .llm_factory import get_chat_model

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
    for attempt in range(3):
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
            if style == "ollama" and (attempt >= 1):
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






