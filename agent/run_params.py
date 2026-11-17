"""本次运行的参数覆盖 (contextvar)。

为什么不用改全局 config: 服务是并发的, 一个人把 rerank 阈值调到 0.9
会悄悄影响另一个人的请求。这里按 asyncio 任务隔离, 只作用于当前 run。

参数键 (前端参数面板 → 后端):
  rag_top_k          检索返回条数 (默认 5)
  rag_rerank         是否启用重排 (默认 True)
  rerank_min_score   重排分数阈值 (默认 config.RERANK_MIN_SCORE = 0)
  simple_max_len     简单任务长度阈值 (默认 60)
  agent_max_rounds   单个团队 Agent 最大轮次 (默认 config.TEAM_AGENT_MAX_ROUNDS)
  agent_timeout      单个团队 Agent 超时秒数 (默认 config.AGENT_AGENT_TIMEOUT)
  parallel_enabled   同层无门 Agent 是否并行 (默认 True)
"""

from contextvars import ContextVar

_PARAMS: ContextVar = ContextVar("run_params", default={})


def set_run_params(params: dict | None):
    return _PARAMS.set(dict(params or {}))


def reset_run_params(token) -> None:
    try:
        _PARAMS.reset(token)
    except Exception:
        pass


def get_param(key: str, default=None):
    """读取本次运行参数; 未设置时返回代码默认值。"""
    try:
        return _PARAMS.get().get(key, default)
    except Exception:
        return default
