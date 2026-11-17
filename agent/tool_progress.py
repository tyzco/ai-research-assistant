"""工具内部进度通道 (contextvar)。

工具是普通函数/方法, 拿不到 agent 的 step_cb。可是"工具正在干什么"恰恰是
最需要透明的一段: 一次 agent_ask 内部有 改写 → 向量/BM25 召回 → RRF →
重排 → 生成。只推 tool_start/tool_end, 前端只能看到"工具在跑", 看不到
RAG 的质量链路。

这里用 contextvar 把当前 run 的推送回调带进工具调用栈, 检索代码在关键
阶段 emit 即可; 没有回调时 (评测/CLI) 是 no-op, 零副作用。
"""

import asyncio
from contextvars import ContextVar

_EMITTER: ContextVar = ContextVar("tool_progress_emitter", default=None)


def set_emitter(cb):
    """在工具执行前设置; 返回 token 供 reset。"""
    return _EMITTER.set(cb)


def reset_emitter(token) -> None:
    try:
        _EMITTER.reset(token)
    except Exception:
        pass


def emit(ev: dict) -> None:
    """同步发一条工具内部事件; 回调返回 coroutine 时挂到当前事件循环。"""
    cb = _EMITTER.get()
    if cb is None:
        return
    try:
        r = cb(ev)
        if asyncio.iscoroutine(r):
            try:
                asyncio.get_running_loop().create_task(r)
            except RuntimeError:
                r.close()
    except Exception:
        # 可视化通道绝不允许影响检索本身
        pass
