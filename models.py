"""Pydantic 数据模型。"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TopicStatus(str, Enum):
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


class TopicState(BaseModel):
    """课题运行时状态。"""

    topic_id: str
    query: str
    status: TopicStatus = TopicStatus.BUILDING
    total_papers: int = 0
    uploaded_papers: int = 0
    total_images: int = 0
    step: str = ""
    error: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    lancedb_table: str = ""
    search_strategy: dict | None = None
    # 进度追踪
    current: int = 0
    dl_total: int = 0
    cn_papers: int = 0
    en_papers: int = 0
    dl_failed: int = 0
    search_strategy: dict | None = None


# ---- Paper (simplified) ----
class PaperMeta(BaseModel):
    model_config = {"extra": "allow"}
    paper_id: str
    title: str
    authors: str = ""
    year: int | None = None
    abstract: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    is_oa: bool = False


# ---- API Request/Response ----
class CreateTopicRequest(BaseModel):
    query: str
    model: str = ""


class CreateTopicResponse(BaseModel):
    topic_id: str
    strategy: dict


class TopicStatusResponse(BaseModel):
    status: str
    progress: str
    step: str = ""


class AskRequest(BaseModel):
    topic_id: str
    question: str
    model: str = ""
    vision_model: str = ""


class AskResponse(BaseModel):
    answer: str
    references: list[dict[str, Any]] = Field(default_factory=list)
    supplement: list[dict[str, Any]] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
