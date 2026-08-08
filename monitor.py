"""监控与安全模块：全链路日志 + 指标收集 + 输入过滤 + API 重试。

用法：
    from monitor import MetricsTracker, safe_call, sanitize_input
    metrics = MetricsTracker()  # 全局单例
    result = await safe_call(llm_func, max_retries=2, timeout=15)
    clean = sanitize_input(user_text)
"""

import asyncio
import logging
import re
import threading
import time
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ===== 安全过滤 =====

from config import INJECTION_PATTERNS

# Unicode 零宽字符 / 混淆字符 (可被用于绕过过滤)
_ZERO_WIDTH_CHARS = re.compile("[​‌‍‎‏⁠⁡⁢⁣⁤﻿￹￺￻]")


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """过滤用户输入：Unicode 规范化 → 移除零宽字符 → 截断 → 防注入。"""
    if not text:
        return ""

    # Unicode 规范化 (NFC): 防止同形异义攻击
    cleaned = unicodedata.normalize("NFC", str(text))

    # 移除 Unicode 零宽字符 (BOM / LRE / RLE / ZWSP 等，可被用于绕过检测)
    cleaned = _ZERO_WIDTH_CHARS.sub("", cleaned)

    # 移除不可见 ASCII 控制字符 (保留 \t \n \r)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)

    # 安全截断 (Python 字符串按 Unicode 码点计数，不会截断多字节字符)
    cleaned = cleaned[:max_length].strip()

    # 检测注入
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            logger.warning(
                f"Potential prompt injection detected, sanitizing: {pattern}"
            )
            cleaned = re.sub(pattern, "[filtered]", cleaned, flags=re.IGNORECASE)

    return cleaned


# ===== 指标收集 =====


class MetricsTracker:
    """全局指标计数器（内存 + threading.Lock，线程安全）。"""

    def __init__(self):
        # RLock (可重入): to_dict() 在持锁时调 estimate_cost() 也加锁, Lock 会自死锁 → /metrics 永久挂起
        self._lock = threading.RLock()
        self.total_requests = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.search_count = 0
        self.download_count = 0
        self.search_errors = 0
        self.llm_errors = 0
        # 细粒度错误分类 (DeepSeek 对话建议)
        self.error_by_type: dict[str, int] = {}
        self.start_time = time.time()

    def record_request(
        self,
        tokens: int = 0,
        latency_ms: float = 0.0,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        with self._lock:
            self.total_requests += 1
            self.total_tokens += tokens
            self.total_latency_ms += latency_ms
            # 记录真实 token 分布 (用于准确成本估算)
            if prompt_tokens or completion_tokens:
                if not hasattr(self, "_prompt_tokens_total"):
                    self._prompt_tokens_total = 0
                    self._completion_tokens_total = 0
                self._prompt_tokens_total += prompt_tokens
                self._completion_tokens_total += completion_tokens

    def record_error(self, source: str = "", error_type: str = ""):
        with self._lock:
            self.error_count += 1
            if source == "search":
                self.search_errors += 1
            elif source == "llm":
                self.llm_errors += 1
            if error_type:
                self.error_by_type[error_type] = (
                    self.error_by_type.get(error_type, 0) + 1
                )

    def record_search(self):
        with self._lock:
            self.search_count += 1

    def record_download(self):
        with self._lock:
            self.download_count += 1

    def estimate_cost(self) -> dict:
        """Token 成本估算：优先使用真实 prompt/completion 分布，否则 50:50。"""
        with self._lock:
            prompt = getattr(self, "_prompt_tokens_total", 0)
            completion = getattr(self, "_completion_tokens_total", 0)
            if prompt > 0 or completion > 0:
                est = prompt / 1e6 * 0.14 + completion / 1e6 * 0.28
            else:
                in_t = self.total_tokens // 2
                out_t = self.total_tokens - in_t
                est = in_t / 1e6 * 0.14 + out_t / 1e6 * 0.28
            return {
                "total_tokens": self.total_tokens,
                "estimated_cost_usd": round(est, 4),
            }

    def to_dict(self) -> dict:
        with self._lock:
            uptime = max(1, time.time() - self.start_time)
            avg_latency = self.total_latency_ms / max(1, self.total_requests)
            error_rate = self.error_count / max(1, self.total_requests)
            return {
                "uptime_seconds": round(uptime),
                "total_requests": self.total_requests,
                "avg_latency_ms": round(avg_latency, 1),
                "error_rate": round(error_rate, 4),
                "total_tokens": self.total_tokens,
                **self.estimate_cost(),
                "search_count": self.search_count,
                "download_count": self.download_count,
                "search_errors": self.search_errors,
                "llm_errors": self.llm_errors,
                "error_by_type": dict(self.error_by_type),
            }


metrics = MetricsTracker()


# ===== API 安全调用包装器 =====


async def safe_call(
    func,
    *args,
    max_retries: int = 2,
    timeout: float = 15.0,
    source: str = "general",
    **kwargs,
) -> Any:
    """包装异步函数调用：超时 + 指数退避重试 + 降级。

    func 必须是异步可调用对象 (async def 或返回 awaitable)。
    """
    last_error = None
    last_error_type = ""

    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            last_error = TimeoutError(f"{source} 超时 ({timeout}s)")
            last_error_type = "timeout"
            logger.warning(f"[Retry {attempt+1}/{max_retries}] {last_error}")
        except Exception as e:
            last_error = e
            last_error_type = type(e).__name__
            logger.warning(f"[Retry {attempt+1}/{max_retries}] {source} error: {e}")

        # 指数退避: 1s, 2s, 4s... 避免立即重试加重下游压力
        if attempt < max_retries:
            backoff = min(2**attempt, 8)  # cap at 8s
            await asyncio.sleep(backoff)

    # 全部重试失败 → 降级
    metrics.record_error(source, error_type=last_error_type)
    logger.error(f"{source} all retries failed ({last_error_type}): {last_error}")

    if source in ("llm",):
        return None
    elif source == "search":
        return []
    return {}
