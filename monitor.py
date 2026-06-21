"""监控与安全模块：全链路日志 + 指标收集 + 输入过滤 + API 重试。

用法：
    from monitor import MetricsTracker, safe_call, sanitize_input
    metrics = MetricsTracker()  # 全局单例
    result = await safe_call(llm_func, max_retries=2, timeout=15)
    clean = sanitize_input(user_text)
"""

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


# ===== 安全过滤 =====

# Prompt 注入防护关键词
INJECTION_PATTERNS = [
    r"忽略.*指令",
    r"ignore.*instruction",
    r"system\s*:",
    r"你是一个",
    r"you are a",
    r"忘记.*规则",
    r"forget.*rule",
    r"切换角色",
    r"switch.*role",
    r"<<SYS>>",
    r"\[INST\]",
    r"\[SYSTEM\]",
    r"DAN\s*:",
    r"jailbreak",
]


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """过滤用户输入：去特殊字符、截断超长、防注入。"""
    if not text:
        return ""
    cleaned = str(text)[:max_length].strip()
    # 移除不可见控制字符
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    # 检测注入
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            logger.warning(
                f"Potential prompt injection detected, sanitizing: {pattern}"
            )
            cleaned = re.sub(pattern, "[filtered]", cleaned, flags=re.IGNORECASE)
    return cleaned


def validate_pdf_header(content: bytes) -> bool:
    """校验 PDF 文件头魔数（%PDF-）。"""
    return len(content) > 10 and content[:5] == b"%PDF-"


def validate_filename(filename: str) -> bool:
    """防止路径遍历攻击。"""
    return not (".." in filename or "/" in filename or "\\" in filename)


# ===== 指标收集 =====


class MetricsTracker:
    """全局指标计数器（内存，可扩展 Redis）。"""

    def __init__(self):
        self.total_requests = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.search_count = 0
        self.download_count = 0
        self.search_errors = 0
        self.llm_errors = 0
        self.start_time = time.time()

    def record_request(self, tokens: int = 0, latency_ms: float = 0.0):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms

    def record_error(self, source: str = ""):
        self.error_count += 1
        if source == "search":
            self.search_errors += 1
        elif source == "llm":
            self.llm_errors += 1

    def record_search(self):
        self.search_count += 1

    def record_download(self):
        self.download_count += 1

    def to_dict(self) -> dict:
        uptime = max(1, time.time() - self.start_time)
        avg_latency = self.total_latency_ms / max(1, self.total_requests)
        error_rate = self.error_count / max(1, self.total_requests)
        return {
            "uptime_seconds": round(uptime),
            "total_requests": self.total_requests,
            "avg_latency_ms": round(avg_latency, 1),
            "error_rate": round(error_rate, 4),
            "total_tokens": self.total_tokens,
            "search_count": self.search_count,
            "download_count": self.download_count,
            "search_errors": self.search_errors,
            "llm_errors": self.llm_errors,
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
    """包装异步函数调用：超时 + 重试 + 降级。"""
    import asyncio

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs) if callable(func) else func,
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            last_error = TimeoutError(f"{source} 超时 ({timeout}s)")
            logger.warning(f"[Retry {attempt+1}/{max_retries}] {last_error}")
        except Exception as e:
            last_error = e
            logger.warning(f"[Retry {attempt+1}/{max_retries}] {source} error: {e}")

    # 全部重试失败 → 降级
    metrics.record_error(source)
    logger.error(f"{source} all retries failed: {last_error}")
    return None if source in ("llm",) else ([] if source == "search" else {})


def safe_sync_call(func, *args, default=None, **kwargs) -> Any:
    """包装同步函数调用：异常捕获 + 降级。"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Sync call failed: {e}")
        return default
