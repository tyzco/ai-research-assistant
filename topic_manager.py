"""课题管理与进度追踪：内存字典 + JSON 持久化。
会话保留：服务重启后自动恢复历史课题列表。"""

import json
import logging
import uuid
from pathlib import Path

from models import TopicState

logger = logging.getLogger(__name__)

TOPICS_FILE = Path("data/topics.json")
TOPICS_FILE.parent.mkdir(parents=True, exist_ok=True)

active_topics: dict[str, TopicState] = {}


def _save_topics():
    """持久化课题列表到 JSON。"""
    data = {}
    for tid, state in active_topics.items():
        data[tid] = {
            "topic_id": tid,
            "query": state.query,
            "status": state.status.value
            if hasattr(state.status, "value")
            else str(state.status),
            "lancedb_table": state.lancedb_table,
            "total_papers": state.total_papers,
            "created_at": state.created_at,
        }
    TOPICS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _load_topics():
    """从磁盘恢复课题列表。"""
    if not TOPICS_FILE.exists():
        return
    try:
        data = json.loads(TOPICS_FILE.read_text())
        for tid, d in data.items():
            state = TopicState(
                topic_id=tid,
                query=d.get("query", ""),
                lancedb_table=d.get("lancedb_table", ""),
                total_papers=d.get("total_papers", 0),
                created_at=d.get("created_at", ""),
            )
            # 恢复状态
            status_str = d.get("status", "building")
            if status_str == "ready":
                state.status = (
                    state.__class__.status.__class__.READY
                    if hasattr(state.status, "__class__")
                    else "ready"
                )
            active_topics[tid] = state
        logger.info(f"Loaded {len(data)} topics from disk")
    except Exception as e:
        logger.warning(f"Failed to load topics: {e}")


def create_topic(query: str) -> TopicState:
    topic_id = uuid.uuid4().hex[:12]
    state = TopicState(topic_id=topic_id, query=query)
    active_topics[topic_id] = state
    _save_topics()
    return state


def save_topic_state(topic_id: str):
    """手动触发持久化（KB 构建完成时调用）。"""
    if topic_id in active_topics:
        _save_topics()


# 启动时自动加载
_load_topics()
