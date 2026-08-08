"""持久技能记忆 (P3) — 跨会话学习: 记录成功任务的做法, 相似任务复用。

数据飞轮第三环: 成功经验 (任务 → 工具序列 + 结果预览) 持久化到 SQLite,
相似任务启动时注入系统提示, 让 agent 复用已验证有效的做法。

轻量设计: 不做 LLM 蒸馏 (P2 教训已覆盖抽象教训), 直接用「工具序列 + 结果预览」
作为"策略", 二元组重叠匹配相似任务 (与 _find_kb_topic 同思路)。
"""

import logging

logger = logging.getLogger(__name__)


def _bigrams(s: str) -> set:
    s = (s or "").lower().replace(" ", "")
    return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set(s)


def record_success(task: str, tools_used: list, result_preview: str = ""):
    """记录一次成功任务的做法 (工具序列 + 结果预览)。"""
    try:
        from database import get_db

        get_db().record_skill_memory(
            task or "",
            ", ".join(tools_used or []),
            result_preview or "",
        )
    except Exception as e:
        logger.warning(f"record_skill_memory failed: {e}")


def get_similar_strategies(task: str, limit: int = 2, min_overlap: float = 0.5):
    """返回与当前任务相似的成功做法 (按二元组重叠率)。空列表 = 无相似经验。"""
    try:
        from database import get_db

        rows = get_db().get_skill_memory(limit=100)
    except Exception as e:
        logger.warning(f"get_skill_memory failed: {e}")
        return []
    if not rows:
        return []
    b1 = _bigrams(task)
    if not b1:
        return []
    scored = []
    for r in rows:
        b2 = _bigrams(r.get("task", ""))
        if not b2:
            continue
        ov = len(b1 & b2) / min(len(b1), len(b2))
        if ov >= min_overlap:
            scored.append((ov, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:limit]]
