"""Unified system prompts for LLM calls.

Previously the project had 20+ different system prompts, which broke
prefix-based prompt caching (DeepSeek/OpenAI cache by matching the beginning
of the message sequence). Different system messages → different cache keys.

Now consolidated into 3 prompts. Specific instructions move to the USER message
so the SYSTEM prefix stays identical across calls.
"""

# ── Primary: used by 80% of calls (generation, extraction, editing) ─────────

SYSTEM_ASSISTANT = "你是一个中文文档处理助手。"

# ── Classification / decision-making (needs stricter output control) ────────

SYSTEM_CLASSIFIER = ("你是一个中文文档处理助手。"
                     "只输出指定的标签、JSON或简短结果，不要任何解释或额外文本。")

# ── Reasoning / multi-step (for CoT, decomposition, synthesis) ──────────────

SYSTEM_REASONING = ("你是一个中文文档处理助手。"
                    "请展示推理过程，然后给出最终答案。")
