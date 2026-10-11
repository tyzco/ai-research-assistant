"""Pydantic 数据模型 — v5.1 重构: SearchStrategy 具体化 + TopicState 适度拆分。

DeepSeek 对话建议:
  - 稳定的核心字段 → 具体化 (SearchStrategy)
  - 高频更新的字段 → 分组 (TopicProgress)
  - 低频统计字段 → 分组 (TopicStats)
  - 实验性/动态字段 → 保留 dict 或 model_config extra
"""

from datetime import datetime
from enum import Enum
from typing import Any

import pydantic
from pydantic import BaseModel, Field


class TopicStatus(str, Enum):
    BUILDING = "building"  # 建库中
    READY = "ready"  # 可用/执行中 (可问答、写作)
    ERROR = "error"
    SETTLED = "settled"  # 已结题 (初稿交付/用户标记完成)
    ABORTED = "aborted"  # 中止 (用户放弃)
    ARCHIVED = "archived"  # 归档 (从活跃列表隐藏, 防未完成课题充斥上下文)


# ===== Search Strategy (具体化 — DeepSeek 对话建议) =====


class SearchStrategy(BaseModel):
    """搜索策略模型。核心字段类型安全, extra=allow 保留 LLM 动态扩展能力。"""

    model_config = {"extra": "allow"}

    keywords_cn: list[str] = Field(default_factory=list, description="中文关键词")
    keywords_en: list[str] = Field(default_factory=list, description="英文关键词")
    domain_tags: list[str] = Field(default_factory=list, description="学科标签")
    related_terms_cn: list[str] = Field(default_factory=list, description="中文联想术语")
    related_terms_en: list[str] = Field(default_factory=list, description="英文联想术语")
    boolean_queries: list[dict] = Field(default_factory=list, description="布尔检索式")
    recommended_databases: list[str] = Field(default_factory=list, description="推荐数据库")
    top_authors: list[dict] = Field(default_factory=list, description="关键学者")
    search_tips: str = ""


# ===== TopicProgress (高频更新字段 — DeepSeek 对话建议拆分) =====


class TopicProgress(BaseModel):
    """下载/构建进度追踪。高频更新，独立分组便于批量操作。"""

    current: int = 0  # 当前处理到第几篇
    dl_total: int = 0  # 总共需要下载的论文数
    cn_papers: int = 0  # 中文论文数
    en_papers: int = 0  # 英文论文数
    dl_failed: int = 0  # 下载失败数
    rejected: int = 0  # 解析质量闸门拦下的文件数 (扫描件/乱码, 未入库)


class TopicStats(BaseModel):
    """论文统计。低频变化，独立分组便于 API 响应裁剪。"""

    total_papers: int = 0
    uploaded_papers: int = 0
    total_images: int = 0


# ===== TopicState (主模型 — 17字段 → 7 平铺 + 4 子模型字段) =====


class TopicState(BaseModel):
    """课题运行时状态。适度拆分，核心字段平铺，进度/统计分组。"""

    # 核心标识 (平铺 — 高频访问)
    topic_id: str
    query: str
    status: TopicStatus = TopicStatus.BUILDING
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: str = ""  # 归属用户 (空 = 匿名/共享, 见 api /topics 隔离)

    # 子模型 (分组 — 职责单一)
    progress: TopicProgress = Field(default_factory=TopicProgress)
    stats: TopicStats = Field(default_factory=TopicStats)

    # 策略 (具体化 + extra)
    search_strategy: SearchStrategy | None = None

    # 其余 (平铺 — 低频或临时)
    step: str = ""
    error: str = ""
    lancedb_table: str = ""

    # ===== 向后兼容桥接 (DEPRECATED — 将在下一个大版本移除) =====
    # 新代码请使用 state.stats.* 和 state.progress.* 直接访问。
    # 这些 property 仅为迁移期兼容，调用点已收敛，计划删除。

    @property
    def total_papers(self) -> int:
        return self.stats.total_papers

    @total_papers.setter
    def total_papers(self, v: int):
        self.stats.total_papers = v

    @property
    def uploaded_papers(self) -> int:
        return self.stats.uploaded_papers

    @uploaded_papers.setter
    def uploaded_papers(self, v: int):
        self.stats.uploaded_papers = v

    @property
    def total_images(self) -> int:
        return self.stats.total_images

    @total_images.setter
    def total_images(self, v: int):
        self.stats.total_images = v

    @property
    def current(self) -> int:
        return self.progress.current

    @current.setter
    def current(self, v: int):
        self.progress.current = v

    @property
    def dl_total(self) -> int:
        return self.progress.dl_total

    @dl_total.setter
    def dl_total(self, v: int):
        self.progress.dl_total = v

    @property
    def cn_papers(self) -> int:
        return self.progress.cn_papers

    @cn_papers.setter
    def cn_papers(self, v: int):
        self.progress.cn_papers = v

    @property
    def en_papers(self) -> int:
        return self.progress.en_papers

    @en_papers.setter
    def en_papers(self, v: int):
        self.progress.en_papers = v

    @property
    def dl_failed(self) -> int:
        return self.progress.dl_failed

    @dl_failed.setter
    def dl_failed(self, v: int):
        self.progress.dl_failed = v


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
    citation_count: int = 0  # 引用数，Semantic Scholar / OpenAlex 提供


# ---- API Request/Response ----
class CreateTopicRequest(BaseModel):
    query: str
    model: str = ""
    topic_id: str = ""
    reuse_kb: bool = False  # 写论文模式: 已有同名/相似且建库的课题则复用, 跳过多源检索


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
    history: list = []


class AskResponse(BaseModel):
    answer: str
    references: list[dict[str, Any]] = Field(default_factory=list)
    supplement: list[dict[str, Any]] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    # 程序化校验结果 (固定结构, 非自由文本): 数字/年份是否能在检索证据中找到。
    # 语义是"证据里有没有这个数", 不是"模型编没编" —— 见 qa_engine.verify_numeric_claims
    numeric_check: dict[str, Any] = Field(default_factory=dict)
    # 召回不足标记: 模型判定库内资料覆盖不到问题主题 (不强行作答)。固定结构而非自由文本,
    # 供前端/监控直接消费
    insufficient: bool = False
    # 引用定位校验: 答案里的"定位"是否真出自本次检索资料 (程序化核验, 零 LLM 调用)
    citation_check: dict[str, Any] = Field(default_factory=dict)
