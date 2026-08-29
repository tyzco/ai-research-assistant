"""课题管理与进度追踪：SQLite 持久化替代 JSON 文件。

服务重启后自动从 SQLite 恢复所有课题。支持多用户隔离。
"""

import json
import logging
import uuid

from models import TopicProgress, TopicState, TopicStats, TopicStatus

logger = logging.getLogger(__name__)

active_topics: dict[str, TopicState] = {}

# 状态反向映射 (DeepSeek 对话建议: 避免 BUILDING 吞掉其他状态)
_STATUS_LOAD_MAP = {
    "ready": TopicStatus.READY,
    "building": TopicStatus.BUILDING,
    "error": TopicStatus.ERROR,
    "settled": TopicStatus.SETTLED,
    "aborted": TopicStatus.ABORTED,
    "archived": TopicStatus.ARCHIVED,
}
_STATUS_MAP_REVERSE = {v.value: v for v in TopicStatus}

# 状态机流转规则: 当前状态 → 允许流转到的状态 (课题生命周期)
# 待立项≈READY(选题报告已出, 可用) → 建库/可用/结题/中止; 结题/中止 → 归档(移出活跃列表)
TRANSITIONS: dict[str, set[str]] = {
    "building": {"ready", "error", "aborted"},
    "error": {"ready", "building", "aborted"},
    "ready": {"settled", "aborted", "archived"},
    "settled": {"archived"},
    "aborted": {"archived"},
}


def transition_topic(topic_id: str, new_status: str, user_id: str = "", reason: str = "") -> tuple[bool, str]:
    """课题状态机流转: 校验合法流转 → 更新内存+持久化 → 审计日志。

    user_id 为权限控制参数 (调用方负责鉴权); 审计日志记录谁/何时/从哪到哪/原因。
    非法流转返回 (False, 原因)。
    """
    state = active_topics.get(topic_id)
    if not state:
        return False, "课题不存在"
    cur = state.status.value
    new_s = _STATUS_MAP_REVERSE.get(new_status)
    if new_s is None:
        return False, f"未知状态: {new_status}"
    if new_status == cur:
        return False, f"课题已是 {cur} 状态"
    allowed = TRANSITIONS.get(cur, set())
    if new_status not in allowed:
        return False, f"非法流转: {cur} → {new_status} (允许: {sorted(allowed) or '无'})"
    old = state.status.value
    state.status = new_s
    save_topic_state(topic_id)
    try:
        from database import get_db

        get_db().log_state_change(topic_id, old, new_status, user_id, reason)
    except Exception as e:
        logger.warning(f"audit log failed: {e}")
    return True, f"{old} → {new_status}"


def _save_topics():
    """全量同步：启动/关闭时调用。"""
    from database import get_db

    db = get_db()
    for tid, state in active_topics.items():
        _save_one(db, tid, state)


def _save_one(db, tid: str, state: TopicState):
    """增量保存单个课题 (DeepSeek 对话建议: 替代全量保存的写放大问题)。"""
    import json as _json

    strategy_json = None
    if state.search_strategy is not None:
        try:
            if hasattr(state.search_strategy, "model_dump"):
                strategy_json = _json.dumps(
                    state.search_strategy.model_dump(), ensure_ascii=False
                )
            else:
                strategy_json = _json.dumps(
                    state.search_strategy, ensure_ascii=False, default=str
                )
        except Exception:
            strategy_json = str(state.search_strategy)[:5000]

    db.save_topic(
        topic_id=tid,
        query=state.query,
        status=state.status.value,
        lancedb_table=state.lancedb_table,
        total_papers=state.stats.total_papers,
        uploaded_papers=state.stats.uploaded_papers,
        total_images=state.stats.total_images,
        search_strategy=strategy_json if strategy_json else None,
        step=state.step,
        current=state.progress.current,
        dl_total=state.progress.dl_total,
        user_id=state.user_id or None,
    )


def _load_topics():
    """从 SQLite 恢复所有课题到内存。"""
    from database import get_db

    db = get_db()
    try:
        rows = db.list_topics_full()
        for d in rows:
            tid = d["topic_id"]
            state = TopicState(
                topic_id=tid,
                query=d.get("query", ""),
                lancedb_table=d.get("lancedb_table", ""),
                user_id=d.get("user_id", "") or "",
                # 显式构造子模型 — Pydantic v2 model_validator 在部分路径下可能丢弃
                stats=TopicStats(
                    total_papers=d.get("total_papers", 0),
                    uploaded_papers=d.get("uploaded_papers", 0),
                    total_images=d.get("total_images", 0),
                ),
                progress=TopicProgress(
                    current=d.get("current", 0),
                    dl_total=d.get("dl_total", 0),
                ),
                step=d.get("step", ""),
                created_at=d.get("created_at", ""),
            )
            # 完整状态映射 (DeepSeek 对话建议: 不丢失 ERROR 等状态)
            status_str = d.get("status", "building")
            state.status = _STATUS_LOAD_MAP.get(status_str, TopicStatus.BUILDING)
            if status_str not in _STATUS_LOAD_MAP:
                logger.warning(
                    f"Unknown topic status '{status_str}' for {tid}, defaulting to BUILDING"
                )

            # JSON 反序列化 (DeepSeek 对话建议: 写入/读取对称)
            strat_raw = d.get("search_strategy")
            if strat_raw:
                try:
                    from models import SearchStrategy

                    strat_dict = (
                        json.loads(strat_raw)
                        if isinstance(strat_raw, str)
                        else strat_raw
                    )
                    state.search_strategy = SearchStrategy(**strat_dict)
                except Exception:
                    logger.debug(
                        f"Could not parse search_strategy for {tid}, using raw"
                    )
                    state.search_strategy = strat_raw

            active_topics[tid] = state
        logger.info(f"Loaded {len(rows)} topics from SQLite")
    except Exception as e:
        logger.warning(f"Failed to load topics from SQLite: {e}")


def create_topic(query: str, topic_id: str = "", user_id: str = None) -> TopicState:
    tid = topic_id if topic_id else uuid.uuid4().hex[:12]
    # 如果 topic 已存在（刷新页面/切换回来），直接返回现有状态，不覆盖
    if tid in active_topics:
        existing = active_topics[tid]
        if query and query != existing.query:
            existing.query = query
            save_topic_state(tid)
        return existing
    state = TopicState(topic_id=tid, query=query, user_id=user_id or "")
    active_topics[tid] = state
    save_topic_state(tid)
    return state


def save_topic_state(topic_id: str):
    """增量保存单个课题 (DeepSeek 对话建议: 原实现名不副实——叫 save_topic_state 却做全量保存)。"""
    if topic_id in active_topics:
        from database import get_db

        _save_one(get_db(), topic_id, active_topics[topic_id])


_load_topics()
