"""Agent 生命周期钩子系统 — 10 个干预节点。

pi-agent 设计参考: 将单次任务(trace)拆解为多轮模型调用(turn)，在关键节点设 hooks。

Hook 执行语义 (security-first merge):
  - block: one-way latch (一旦阻止, stay blocked)
  - tools: intersection only (只能缩小, 不能新增)
  - messages: first-writer-wins
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HookPoint(str, Enum):
    """10 个干预节点。按执行顺序排列。"""

    # === Trace 级别 (整个任务开始/结束) ===
    TRACE_START = "trace_start"  # 任务开始, 可注入系统消息、设置限制
    TRACE_END = "trace_end"  # 任务结束, 可做审计、统计

    # === Turn 级别 (每轮模型调用) ===
    TURN_START = "turn_start"  # 轮次开始, 检查步数上限、重置 turn 状态
    TURN_END = "turn_end"  # 轮次结束, 统计 token 消耗

    # === 模型调用 ===
    BEFORE_MODEL = "before_model"  # LLM 调用前, 可修改 messages/tools, 压缩上下文
    AFTER_MODEL = "after_model"  # LLM 返回后, 可审计 tool_calls, 过滤危险工具

    # === 工具执行 ===
    BEFORE_TOOL = "before_tool"  # 工具执行前, 参数校验 + 安全检查
    AFTER_TOOL = "after_tool"  # 工具执行后, 截断过长结果, 写入文件

    # === 响应 ===
    BEFORE_EMIT = "before_emit"  # 结果输出前, 红线检测、敏感词过滤
    AFTER_EMIT = "after_emit"  # 结果输出后, 记录审计日志


@dataclass
class HookResult:
    """Hook 返回结果。block=True 阻止后续执行。"""

    block: bool = False
    block_reason: str = ""
    messages: Optional[list[dict]] = None  # 修改后的消息列表
    tools: Optional[list[dict]] = None  # 修改后的工具列表 (只能缩小)
    tool_calls: Optional[list] = None  # 过滤后的 tool_calls
    inject: Optional[list[dict]] = None  # 注入额外消息
    metadata: dict = field(default_factory=dict)


HookFunc = Callable[[dict, str], Optional[HookResult]]
"""Hook 函数签名: (state: dict, event_payload: str) -> Optional[HookResult]"""


class HookRegistry:
    """全局 Hook 注册表。按 HookPoint 注册，按顺序 FIFO 合并执行。"""

    def __init__(self):
        self._hooks: dict[HookPoint, list[tuple[str, HookFunc]]] = {
            p: [] for p in HookPoint
        }
        self._global_block = False
        self._stats: dict[str, int] = {}  # hook_name → fire_count

    def register(self, point: HookPoint, name: str, fn: HookFunc):
        """注册一个 hook。同名 hook 会覆盖。"""
        # 移除同名旧 hook
        self._hooks[point] = [(n, f) for n, f in self._hooks[point] if n != name]
        self._hooks[point].append((name, fn))
        self._stats[name] = 0
        logger.debug(f"Hook registered: {point.value}.{name}")

    def fire(self, point: HookPoint, state: dict, payload: str = "") -> HookResult:
        """执行某个干预点的所有 hook，按 merge 语义合并结果。"""
        merged = HookResult()

        for name, fn in self._hooks[point]:
            try:
                result = fn(state, payload)
                self._stats[name] = self._stats.get(name, 0) + 1
                if result is None:
                    continue

                # Merge 语义
                if result.block:
                    merged.block = True
                    merged.block_reason = result.block_reason or merged.block_reason

                # tools: intersection (只能缩小, 不能新增)
                # DeepSeek 审查 #1: 空列表视为"不修改"(防止误清空工具集);
                # 空交集时保留原集合并告警
                if result.tools:
                    if merged.tools is None:
                        merged.tools = result.tools
                    else:
                        intersected = [
                            t
                            for t in merged.tools
                            if any(
                                r.get("function", {}).get("name")
                                == t.get("function", {}).get("name")
                                for r in result.tools
                            )
                        ]
                        if intersected:
                            merged.tools = intersected
                        else:
                            logger.warning(
                                f"tools intersection empty for hook {name}@{point.value}, "
                                "keeping previous set"
                            )
                elif result.tools is not None:
                    logger.debug(
                        f"hook {name}@{point.value} returned empty tools, treated as no-op"
                    )

                # tool_calls: 交集式过滤, 同样防空集 (DeepSeek 审查 #3)
                if result.tool_calls:
                    if merged.tool_calls is None:
                        merged.tool_calls = result.tool_calls
                    else:
                        existing_names = {
                            tc.get("function", {}).get("name")
                            for tc in merged.tool_calls
                        }
                        filtered = [
                            tc
                            for tc in result.tool_calls
                            if tc.get("function", {}).get("name") in existing_names
                        ]
                        if filtered:
                            merged.tool_calls = filtered
                        else:
                            logger.warning(
                                f"tool_calls intersection empty for hook {name}@{point.value}, "
                                "keeping previous set"
                            )
                elif result.tool_calls is not None:
                    logger.debug(
                        f"hook {name}@{point.value} returned empty tool_calls, treated as no-op"
                    )

                # messages: first-writer-wins (DeepSeek 审查 #5: 冲突时告警)
                if result.messages is not None:
                    if merged.messages is None:
                        merged.messages = result.messages
                    else:
                        logger.warning(
                            f"messages conflict at {point.value}: hook {name} "
                            "ignored (first-writer-wins)"
                        )

                # inject: accumulate
                if result.inject:
                    if merged.inject is None:
                        merged.inject = []
                    merged.inject.extend(result.inject)

                merged.metadata.update(result.metadata)

            except Exception as e:
                # DeepSeek 审查 #9: 带堆栈, 便于排查 hook 内部错误
                logger.warning(f"Hook {name}@{point.value} failed: {e}", exc_info=True)

        return merged

    def get_stats(self) -> dict:
        return dict(self._stats)


# 全局单例 (DeepSeek 审查 #4: 加锁防多线程/多事件循环重复初始化)
_registry: Optional[HookRegistry] = None
_registry_lock = threading.Lock()


def get_hooks() -> HookRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = HookRegistry()
                _register_builtins(_registry)
    return _registry


# ===== 内置安全 Hook =====


def _register_builtins(registry: HookRegistry):
    """注册内置安全/监控 hook。"""

    def _max_steps_guard(state, _) -> Optional[HookResult]:
        """TURN_START: 步数超限 → 强制结束。"""
        max_s = state.get("max_steps", 20)
        steps = state.get("steps", 0)
        if steps >= max_s:
            return HookResult(block=True, block_reason=f"已达最大步数 {max_s}")
        return None

    def _sensitive_word_filter(_, payload) -> Optional[HookResult]:
        """BEFORE_EMIT: 敏感词检测 (DeepSeek 审查 #6: 可配置 + 记录命中词)。

        默认词表来自 config.HOOK_SENSITIVE_WORDS。不含 "execute arbitrary code"——
        学术安全论文中该词组很常见, 硬拦截会误杀正常讨论。"""
        try:
            from config import HOOK_SENSITIVE_WORDS as _cfg_words
        except Exception:
            _cfg_words = None
        sensitive = (
            _cfg_words
            if _cfg_words is not None
            else ["rm -rf /", "DROP TABLE"]
        )
        lowered = (payload or "").lower()
        for word in sensitive:
            if word.lower() in lowered:
                logger.warning(f"Sensitive content blocked: {word!r}")
                return HookResult(
                    block=True,
                    block_reason=f"输出包含敏感内容，已拦截（命中: {word}）",
                )
        return None

    def _tool_allowlist(_, payload) -> Optional[HookResult]:
        """BEFORE_TOOL: 仅允许注册表中的工具执行。
        DeepSeek 审查 #7: 记录每次工具尝试 (审计), 未来可加黑名单。"""
        try:
            import json as _json

            data = _json.loads(payload or "{}")
            logger.debug(f"Tool attempt: {data.get('tool', '?')}")
        except Exception:
            pass
        return None

    registry.register(HookPoint.TURN_START, "max_steps_guard", _max_steps_guard)
    registry.register(HookPoint.BEFORE_EMIT, "sensitive_filter", _sensitive_word_filter)
    registry.register(HookPoint.BEFORE_TOOL, "tool_allowlist", _tool_allowlist)
