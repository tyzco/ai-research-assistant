"""LLM 客户端容错封装 — 超时 + 指数退避重试 + 可选降级模型 (企业级第一条)。

问题: 项目内 38 处 `AsyncOpenAI(...)` 裸调, 端点一抖 (超时/5xx/限流) 即整体失败。
方案:
  1. 单例客户端 (带超时, 复用连接池) — get_client()
  2. chat_with_retry() — 主模型 N 次退避重试 → 降级模型 (若配置) 再 N 次 → 才抛异常

降级链 (三者缺一即跳过降级, 仅重试):
  DEEPSEEK_MODEL → (失败) → DEEPSEEK_FALLBACK_MODEL @ DEEPSEEK_FALLBACK_BASE_URL

用法 (与裸 create 完全一致, 返回同一响应对象):
  from llm_client import chat_with_retry
  resp = await chat_with_retry(model=DEEPSEEK_MODEL, messages=msgs, tools=..., ...)
"""
import asyncio
import logging
import threading

from openai import AsyncOpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_FALLBACK_API_KEY,
    DEEPSEEK_FALLBACK_BASE_URL,
    DEEPSEEK_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None
_lock = threading.Lock()


def get_client() -> AsyncOpenAI:
    """单例主客户端 (带超时)。所有调用点共享连接池, 避免重复握手。"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = AsyncOpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    timeout=LLM_TIMEOUT,
                )
    return _client


def _fallback_client() -> AsyncOpenAI | None:
    """降级客户端 (仅当配置了 fallback base_url 才构造)。"""
    if not DEEPSEEK_FALLBACK_BASE_URL:
        return None
    return AsyncOpenAI(
        api_key=DEEPSEEK_FALLBACK_API_KEY or DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_FALLBACK_BASE_URL,
        timeout=LLM_TIMEOUT,
    )


async def _call(client: AsyncOpenAI, *, model: str, messages: list[dict], **kwargs):
    return await client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )


async def chat_with_retry(
    *, model: str, messages: list[dict], retries: int | None = None, **kwargs
):
    """带重试 + 降级的 LLM 调用。kwargs 透传 (tools/tool_choice/temperature/max_tokens/extra_body/stream...)。

    返回 chat.completions 响应对象 (与裸 create 一致), 故 resp.choices / resp.usage 均可用。
    """
    max_retries = LLM_MAX_RETRIES if retries is None else retries
    last: Exception | None = None

    # 主模型退避重试
    for attempt in range(max_retries + 1):
        try:
            return await _call(get_client(), model=model, messages=messages, **kwargs)
        except Exception as e:  # noqa: BLE001 — 降级层需兜住一切瞬时错误
            last = e
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))

    # 降级模型 (三项 fallback 配置齐全才启用)
    fb = _fallback_client()
    if fb and DEEPSEEK_FALLBACK_MODEL:
        logger.warning(f"主模型 {model} 失败, 降级到 {DEEPSEEK_FALLBACK_MODEL}")
        for attempt in range(max_retries + 1):
            try:
                return await _call(
                    fb, model=DEEPSEEK_FALLBACK_MODEL, messages=messages, **kwargs
                )
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

    raise (last if last is not None else RuntimeError("LLM call failed"))
