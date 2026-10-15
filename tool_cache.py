"""工具结果缓存 — 对**只读且同参同果**的工具复用结果。

对应优化项"高频工具结果添加缓存"。为什么值得做:
  一个 agent 任务里同一个查询被重复发起是常态 (换措辞再问一遍知识库、反复确认
  课题状态)。每次都是网络往返 + 可能的 LLM 调用, 而这些调用在同一分钟内答案不会变。

安全性来自**显式声明**: 只有 Tool.cacheable=True 的工具才进缓存。这个字段不能从
kind/retryable 推断 —— download_and_index 与 export_report 都是
"deterministic + retryable", 但一个有副作用一个写文件, 缓存它们会让第二次调用
直接命中缓存而**根本不执行**。

TTL 是必须的而不是可选: 缓存的语义是"短时间内答案不变", 不是"永远不变"。
超出 TTL 自动失效, 避免知识库更新后仍返回旧结果。
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_lock = threading.Lock()
_stats = {"hit": 0, "miss": 0, "evict": 0}


def _key(tool_name: str, args: dict) -> str:
    """缓存键: 工具名 + **规范化参数** (sort_keys 保证 {a,b} 与 {b,a} 同键)。"""
    try:
        blob = json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        blob = str(sorted((args or {}).items()))
    return f"{tool_name}\x00{hashlib.md5(blob.encode('utf-8')).hexdigest()}"


def get(tool_name: str, args: dict):
    """取缓存。未命中/已过期 → None。"""
    from config import TOOL_CACHE_ENABLED, TOOL_CACHE_TTL

    if not TOOL_CACHE_ENABLED:
        return None
    k = _key(tool_name, args)
    now = time.time()
    with _lock:
        item = _cache.get(k)
        if item is None:
            _stats["miss"] += 1
            return None
        ts, val = item
        if now - ts > TOOL_CACHE_TTL:
            _cache.pop(k, None)
            _stats["miss"] += 1
            return None
        _cache.move_to_end(k)
        _stats["hit"] += 1
        return val


def put(tool_name: str, args: dict, value: str) -> None:
    """写缓存 (只存成功的非空结果 —— 缓存错误会让下一次调用拿到旧的失败)。"""
    from config import TOOL_CACHE_ENABLED, TOOL_CACHE_MAX, TOOL_CACHE_TTL

    if not TOOL_CACHE_ENABLED or not value:
        return
    if len(value) > 200_000:  # 超大结果不值得占内存, 交给调用方每次重算
        return
    k = _key(tool_name, args)
    with _lock:
        _cache[k] = (time.time(), value)
        _cache.move_to_end(k)
        while len(_cache) > TOOL_CACHE_MAX:
            _cache.popitem(last=False)
            _stats["evict"] += 1


def stats() -> dict:
    with _lock:
        h, m = _stats["hit"], _stats["miss"]
        return {
            "hit": h,
            "miss": m,
            "size": len(_cache),
            "evict": _stats["evict"],
            "hit_rate": round(h / (h + m), 4) if (h + m) else 0.0,
        }


def clear() -> None:
    with _lock:
        _cache.clear()
        _stats.update({"hit": 0, "miss": 0, "evict": 0})
