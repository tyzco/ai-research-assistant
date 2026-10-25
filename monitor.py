"""监控与安全模块：全链路日志 + 指标收集 + 输入过滤 + API 重试。

用法：
    from monitor import MetricsTracker, safe_call, sanitize_input
    metrics = MetricsTracker()  # 全局单例
    result = await safe_call(llm_func, max_retries=2, timeout=15)
    clean = sanitize_input(user_text)
"""

import asyncio
import collections
import logging
import re
import threading
import time
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ===== 安全过滤 =====

from config import INJECTION_PATTERNS, LATENCY_SAMPLE_SIZE

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
        # 延迟分位数: avg 会把长尾藏起来 —— 一次 30s 的超时混在 100 次 1s 里,
        # 平均值几乎看不出来, 但对用户就是"偶发卡死"。面试/复盘里真正看的是 P95/P99。
        # 用固定窗口的环形缓冲, 内存有界 (不随请求数增长)。
        self._latencies = collections.deque(maxlen=LATENCY_SAMPLE_SIZE)
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
            if latency_ms > 0:
                self._latencies.append(float(latency_ms))
            # 记录真实 token 分布 (用于准确成本估算)
            if prompt_tokens or completion_tokens:
                if not hasattr(self, "_prompt_tokens_total"):
                    self._prompt_tokens_total = 0
                    self._completion_tokens_total = 0
                self._prompt_tokens_total += prompt_tokens
                self._completion_tokens_total += completion_tokens

    def _latency_percentiles(self) -> dict:
        """P50/P95/P99 (ms)。样本不足时返回 None, 不拿 3 个请求编一个分位数。"""
        with self._lock:
            vals = sorted(self._latencies)
        if not vals:
            return {"p50_latency_ms": None, "p95_latency_ms": None, "p99_latency_ms": None, "latency_samples": 0}

        def _pct(q: float) -> float:
            # 最近秩法: 取第 ceil(q*n) 个, 不插值 (插值在大延迟样本上会给出不存在的值)
            import math

            idx = max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))
            return round(vals[idx], 1)

        return {
            "p50_latency_ms": _pct(0.50),
            "p95_latency_ms": _pct(0.95),
            "p99_latency_ms": _pct(0.99),
            "latency_samples": len(vals),
        }

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

    def record_ingest(
        self, ok: bool, pages: int = 0, chars: int = 0, chunks: int = 0
    ):
        """记录一次文档接入 (解析+分块) 结果 — 补齐"接入/索引"环节的量化指标。

        之前全流程只有检索/生成有指标, 接入层是黑的: 解析失败率、平均块大小无从得知,
        出了问题只能猜是"PDF 有问题"还是"分块有问题"。
        """
        with self._lock:
            if not hasattr(self, "_ingest"):
                self._ingest = {
                    "attempts": 0, "ok": 0, "fail": 0,
                    "pages": 0, "chars": 0, "chunks": 0,
                }
            g = self._ingest
            g["attempts"] += 1
            g["ok" if ok else "fail"] += 1
            g["pages"] += pages
            g["chars"] += chars
            g["chunks"] += chunks

    def ingest_stats(self) -> dict:
        """接入/索引环节指标: 解析成功率 + 分块质量 (平均块大小/篇均块数)。"""
        with self._lock:
            g = dict(getattr(self, "_ingest", {}))
        if not g or not g.get("attempts"):
            return {"attempts": 0, "note": "暂无接入记录"}
        ok = g.get("ok", 0)
        chunks = g.get("chunks", 0)
        return {
            "attempts": g["attempts"],
            "parse_ok": ok,
            "parse_fail": g.get("fail", 0),
            "parse_success_rate": round(ok / g["attempts"], 4),
            "total_pages": g.get("pages", 0),
            "total_chars": g.get("chars", 0),
            "total_chunks": chunks,
            "avg_chunk_chars": int(g.get("chars", 0) / chunks) if chunks else 0,
            "avg_chunks_per_paper": round(chunks / ok, 1) if ok else 0,
        }

    def record_degradation(self, path: str):
        """记录一次降级触发 (按路径计数)。

        用途: 降级是保底, 不是正常通道。若某路径计数持续增长 = 正常路径坏了, 该修 try 块
        (本项目已因此类 bug 踩坑 5 次)。/metrics 暴露各路径计数, 一眼看出哪条总在降级。
        """
        with self._lock:
            if not hasattr(self, "_degradations"):
                self._degradations: dict = {}
            self._degradations[path] = self._degradations.get(path, 0) + 1

    def degradation_stats(self) -> dict:
        with self._lock:
            return dict(getattr(self, "_degradations", {}))

    def record_search(self):
        with self._lock:
            self.search_count += 1

    def record_download(self):
        with self._lock:
            self.download_count += 1

    def estimate_cost(self) -> dict:
        """Token 成本估算: 用**配置的真实单价**算实际花费。

        修复: 原实现硬编码 $0.14/$0.28 每 MTok (旧 deepseek-chat 价), 换模型后即失真。
        现在从 config 读单价; **未配置 (0) 时返回 None 而非编一个数** —— 宁可不显示金额,
        也不显示不准的金额。
        """
        from config import (
            DEEPSEEK_PRICE_INPUT_PER_MTOK,
            DEEPSEEK_PRICE_OUTPUT_PER_MTOK,
        )

        with self._lock:
            prompt = getattr(self, "_prompt_tokens_total", 0)
            completion = getattr(self, "_completion_tokens_total", 0)
            if prompt > 0 or completion > 0:
                in_t, out_t = prompt, completion
            else:
                in_t = self.total_tokens // 2
                out_t = self.total_tokens - in_t
            priced = (
                DEEPSEEK_PRICE_INPUT_PER_MTOK > 0
                or DEEPSEEK_PRICE_OUTPUT_PER_MTOK > 0
            )
            cost = None
            if priced:
                cost = (
                    in_t / 1e6 * DEEPSEEK_PRICE_INPUT_PER_MTOK
                    + out_t / 1e6 * DEEPSEEK_PRICE_OUTPUT_PER_MTOK
                )
            return {
                "total_tokens": self.total_tokens,
                "prompt_tokens": in_t,
                "completion_tokens": out_t,
                "estimated_cost": round(cost, 4) if cost is not None else None,
                "cost_note": (
                    ""
                    if priced
                    else "未配置单价 (DEEPSEEK_PRICE_INPUT/OUTPUT_PER_MTOK) → 只报 token, 不报金额"
                ),
            }

    def estimate_calls_cost(self, n_calls: int) -> dict | None:
        """按**历史真实均值**估算 n 次 LLM 调用的成本 (供成本闸事前估算)。

        返回 None 表示数据不足 (无 token 历史) 或未配置单价 → 调用方应降级为只报次数。
        """
        from config import (
            DEEPSEEK_PRICE_INPUT_PER_MTOK,
            DEEPSEEK_PRICE_OUTPUT_PER_MTOK,
        )

        with self._lock:
            reqs = getattr(self, "_total_requests", 0)
            toks = self.total_tokens
            if reqs < 5 or toks <= 0:
                return None  # 历史太少, 均值不可信
            avg_tokens = toks / reqs
        priced = DEEPSEEK_PRICE_INPUT_PER_MTOK > 0 or DEEPSEEK_PRICE_OUTPUT_PER_MTOK > 0
        if not priced:
            return {"calls": n_calls, "avg_tokens_per_call": int(avg_tokens), "cost": None}
        # LLM 调用以输出 token 为主, 按 40/60 粗分 (来自真实总量, 非拍脑袋单价)
        out_t = avg_tokens * 0.6
        in_t = avg_tokens * 0.4
        per_call = (
            in_t / 1e6 * DEEPSEEK_PRICE_INPUT_PER_MTOK
            + out_t / 1e6 * DEEPSEEK_PRICE_OUTPUT_PER_MTOK
        )
        return {
            "calls": n_calls,
            "avg_tokens_per_call": int(avg_tokens),
            "cost": round(per_call * n_calls, 4),
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
                **self._latency_percentiles(),
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
