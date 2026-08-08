"""主动路由的自我进化 (v4): 用建议接受数据优化下一步预判。

数据飞轮: 用户点击建议的「执行」→ 记录 (stage, action) → /suggest 用历史接受度
排序候选建议 → 优先推用户更常接受的动作。这就是「知用户」的路由进化。

存储: data/suggestion_stats.json {stage: {action: count}}
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_stats_file = Path("data/suggestion_stats.json")
_stats: dict = {}


def _load() -> None:
    global _stats
    try:
        if _stats_file.exists():
            _stats = json.loads(_stats_file.read_text())
    except Exception:
        _stats = {}


def _save() -> None:
    try:
        _stats_file.parent.mkdir(parents=True, exist_ok=True)
        _stats_file.write_text(json.dumps(_stats, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"suggestion stats save failed: {e}")


def record_accept(stage: str, action: str) -> None:
    """记录用户接受了某阶段的某建议 (点击执行)。"""
    _load()
    stage_stats = _stats.setdefault(stage, {})
    stage_stats[action] = int(stage_stats.get(action, 0)) + 1
    _save()


def get_stats() -> dict:
    """各阶段建议被接受次数 (主动路由学习观测)。"""
    _load()
    return {stage: dict(actions) for stage, actions in _stats.items()}


def rank_candidates(stage: str, candidates: list[dict]) -> list[dict]:
    """按历史接受度排序候选建议 (接受多者优先; 无数据时保持默认顺序)。"""
    _load()
    counts = _stats.get(stage, {})
    # 附加接受次数, 降序; 同分保持默认顺序
    ranked = sorted(
        enumerate(candidates),
        key=lambda iv: -int(counts.get(iv[1]["action"], 0)),
    )
    return [c for _, c in ranked]


_load()
