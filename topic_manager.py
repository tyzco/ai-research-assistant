"""课题管理与进度追踪：内存字典存储运行时状态。"""

import logging
import uuid

from models import TopicState

logger = logging.getLogger(__name__)

active_topics: dict[str, TopicState] = {}


def create_topic(query: str) -> TopicState:
    topic_id = uuid.uuid4().hex[:12]
    state = TopicState(topic_id=topic_id, query=query)
    active_topics[topic_id] = state
    return state
