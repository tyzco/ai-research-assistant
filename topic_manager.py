"""课题管理与进度追踪：内存字典 + JSON 持久化。
会话保留：服务重启后自动恢复历史课题列表。"""

import json
import logging
import uuid
from pathlib import Path

from models import TopicState, TopicStatus

logger = logging.getLogger(__name__)

TOPICS_FILE = Path("data/topics.json")
TOPICS_FILE.parent.mkdir(parents=True, exist_ok=True)

active_topics: dict[str, TopicState] = {}


def _save_topics():
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
            status_str = d.get("status", "building")
            state.status = (
                TopicStatus.READY if status_str == "ready" else TopicStatus.BUILDING
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
    if topic_id in active_topics:
        _save_topics()


_load_topics()
