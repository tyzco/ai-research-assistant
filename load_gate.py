"""质量水位管控 (load-aware degradation): 按实时并发负载动态升降检索质量档位。

对应分布式系统的"一致性水位管控"在 RAG 系统的正确形态——
不是"要强一致还是最终一致"的静态选择, 而是**按实时负载动态升降质量档位**:

  低水位 (low)      并发充足 → 全流程: 多变体召回 + 两级级联重排 + HyDE(若开) → 最高质量
  高水位 (high)     并发偏高 → 降级: 跳两级级联的 stage2 精排(只用 stage1) + 跳查询变体 → 保速度
  超水位 (overload) 并发过载 → 极致降级: 单路召回 + 单级重排 → 保可用 (连接层 limit_concurrency 兜底)

与"失败降级"(异常才触发)的区别: 水位降级是**负载触发的主动预防** — 高并发下主动放弃部分质量
换取整体不溢出, 负载回落后自动升回满档。

用法:
    from load_gate import enter, exit, load_level
    enter()          # 请求进入
    ...
    exit()           # 请求退出
    load_level()     # "low" | "high" | "overload"
"""
import logging
import threading

logger = logging.getLogger(__name__)

_in_flight = 0
_lock = threading.Lock()
_peak = 0  # 观测用: 进程内峰值并发


def _thresholds() -> tuple[int, int]:
    from config import LOAD_HIGH_THRESHOLD, LOAD_OVERLOAD_THRESHOLD

    return LOAD_HIGH_THRESHOLD, LOAD_OVERLOAD_THRESHOLD


def enter() -> None:
    """请求进入 (计数 +1)。"""
    global _in_flight, _peak
    with _lock:
        _in_flight += 1
        if _in_flight > _peak:
            _peak = _in_flight


def exit() -> None:
    """请求退出 (计数 -1)。"""
    global _in_flight
    with _lock:
        if _in_flight > 0:
            _in_flight -= 1


def in_flight() -> int:
    with _lock:
        return _in_flight


def peak() -> int:
    with _lock:
        return _peak


def load_level() -> str:
    """当前水位: low / high / overload。"""
    high, overload = _thresholds()
    n = in_flight()
    if n >= overload:
        return "overload"
    if n >= high:
        return "high"
    return "low"


def stats() -> dict:
    """观测统计 (供 /metrics)。"""
    return {"in_flight": in_flight(), "peak": peak(), "level": load_level()}
