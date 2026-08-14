"""配置管理：所有可调参数集中于此。其他模块通过 from config import XXX 引用。

约定: 可调旋钮 (数值/字符串/列表) 一律 env 驱动 + 默认值; 内容型数据
(意图关键词/垃圾词/注入模式/翻译字典) 作为代码常驻 — 那是内容, 不是配置旋钮。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# HF/Transformers 强制离线: 所有模型本地缓存, 禁止联网检查。
# 修复: 以前只在 embedder.py 设 HF_HUB_OFFLINE, 任何不先 import embedder 的模型加载
# (如 reranker) 都会尝试连 huggingface.co → 本机网络抖, 挂起 30-60s。config 被所有模块先导入,
# 在此全局设默认值 (尊重用户显式 env 覆盖)。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _env(name: str, default, cast=str):
    """读环境变量; 缺失/非法时回退默认。统一 cast, 消除 os.getenv 样板。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, default, lambda v: str(v).lower() in ("1", "true", "yes", "on"))


def _env_list(name: str, default: list):
    """读逗号分隔环境变量为列表。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


# ===================================================================
# LLM (DeepSeek)
# ===================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-v4-pro")

# LLM 容错 (企业级第一条: 单点故障容错)。单 DeepSeek 端点抖动即失败 →
# 超时 + 指数退避重试 + 可选降级模型。fallback 三项留空 = 不降级 (仅重试)。
LLM_TIMEOUT = _env("LLM_TIMEOUT", 60, int)
LLM_MAX_RETRIES = _env("LLM_MAX_RETRIES", 2, int)
DEEPSEEK_FALLBACK_MODEL = _env("DEEPSEEK_FALLBACK_MODEL", "", str)
DEEPSEEK_FALLBACK_BASE_URL = _env("DEEPSEEK_FALLBACK_BASE_URL", "", str)
DEEPSEEK_FALLBACK_API_KEY = os.getenv("DEEPSEEK_FALLBACK_API_KEY", "")
# 选题判断 (钩子) 用 pro 模型保质量 (第一印象, 值得花 pro 的钱; flash 省钱给后台/推送)
RESEARCH_GAP_MODEL = _env("RESEARCH_GAP_MODEL", "deepseek-v4-pro")

# ===================================================================
# Embedding (本地模型)
# ===================================================================
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = _env("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
LOCAL_EMBEDDING_DIM = _env("LOCAL_EMBEDDING_DIM", 512, int)
# 嵌入后端: "sentence_transformers" | "flag_embedding" | "" (留空按模型名自动检测)
EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "")

# ===================================================================
# Rerank 模型 (延迟加载)
# ===================================================================
RERANK_MODEL = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_STAGE1_MODEL = _env("RERANK_STAGE1_MODEL", "all-MiniLM-L6-v2")

# ===================================================================
# Vision Model (默认关闭)
# ===================================================================
ENABLE_VISION = _env_bool("ENABLE_VISION", False)
VISION_MODEL = _env("VISION_MODEL", "qwen-vl-max")
# 学术范围白名单闸门 (锁定用户只能问论文/学术相关问题, 默认开启)
SCOPE_GATE_ENABLED = _env_bool("SCOPE_GATE_ENABLED", True)
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_BASE_URL = _env(
    "VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ===================================================================
# Academic Search
# ===================================================================
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")
MAX_SEARCH_RESULTS = _env("MAX_SEARCH_RESULTS", 50, int)

# ===================================================================
# Storage Paths
# ===================================================================
PAPER_CACHE_DIR = PROJECT_ROOT / os.getenv("PAPER_CACHE_DIR", "paper_cache")
LANCEDB_DIR = PROJECT_ROOT / os.getenv("LANCEDB_DIR", "data/lancedb_data")
IMAGE_DIR = PROJECT_ROOT / os.getenv("IMAGE_DIR", "data/images")

# ===================================================================
# Retrieval Limits
# ===================================================================
# 分块大小 (字符): 论文长, 2048 太小 → 检索到的段落覆盖不全 (答案跨块会丢)。
# 4096 让每个检索单元覆盖更全; 配合 1M 上下文窗口, prompt 可容纳整块。
CHUNK_SIZE = _env("CHUNK_SIZE", 4096, int)
RETRIEVAL_TOP_K = _env("RETRIEVAL_TOP_K", 5, int)
MAX_CONTEXT_LENGTH = _env("MAX_CONTEXT_LENGTH", 300, int)  # chars per chunk in prompt

# ===================================================================
# LLM Call Parameters (all LLM calls use these defaults)
# ===================================================================
# Generation (answers, reports)
LLM_GEN_TEMPERATURE = _env("LLM_GEN_TEMPERATURE", 0.3, float)
# deepseek-v4-pro 是推理模型, max_tokens 同时计入推理+输出; 1024 在复杂 RAG prompt 下
# 推理可能耗尽预算 → 输出空串。提高兜底 (实际输出已在推理禁用后显著小于此值)。
LLM_GEN_MAX_TOKENS = _env("LLM_GEN_MAX_TOKENS", 2048, int)
LLM_GEN_TIMEOUT = _env("LLM_GEN_TIMEOUT", 30, int)

# Quick responses (agent/quick, translations)
LLM_QUICK_TEMPERATURE = _env("LLM_QUICK_TEMPERATURE", 0.1, float)
LLM_QUICK_MAX_TOKENS = _env("LLM_QUICK_MAX_TOKENS", 512, int)
LLM_QUICK_TIMEOUT = _env("LLM_QUICK_TIMEOUT", 15, int)

# Strategy generation (search strategy, entity extraction)
LLM_STRATEGY_TEMPERATURE = _env("LLM_STRATEGY_TEMPERATURE", 0.3, float)
LLM_STRATEGY_MAX_TOKENS = _env("LLM_STRATEGY_MAX_TOKENS", 2000, int)
LLM_STRATEGY_TIMEOUT = _env("LLM_STRATEGY_TIMEOUT", 30, int)

# Translation (CN→EN keyword translation)
LLM_TRANSLATE_TEMPERATURE = _env("LLM_TRANSLATE_TEMPERATURE", 0.0, float)
LLM_TRANSLATE_MAX_TOKENS = _env("LLM_TRANSLATE_MAX_TOKENS", 50, int)
LLM_TRANSLATE_TIMEOUT = _env("LLM_TRANSLATE_TIMEOUT", 6, int)

# Coreference resolution (multi-turn)
LLM_COREF_TEMPERATURE = _env("LLM_COREF_TEMPERATURE", 0.1, float)
LLM_COREF_MAX_TOKENS = _env("LLM_COREF_MAX_TOKENS", 100, int)
LLM_COREF_TIMEOUT = _env("LLM_COREF_TIMEOUT", 8, int)

# Query expansion
LLM_EXPAND_TEMPERATURE = _env("LLM_EXPAND_TEMPERATURE", 0.3, float)
LLM_EXPAND_MAX_TOKENS = _env("LLM_EXPAND_MAX_TOKENS", 200, int)
LLM_EXPAND_TIMEOUT = _env("LLM_EXPAND_TIMEOUT", 10, int)

# ===================================================================
# Search Timeouts (per source, seconds)
# ===================================================================
SEARCH_TIMEOUT_ARXIV = _env("SEARCH_TIMEOUT_ARXIV", 10, int)
SEARCH_TIMEOUT_OPENALEX = _env("SEARCH_TIMEOUT_OPENALEX", 10, int)
SEARCH_TIMEOUT_OPENALEX_CN = _env("SEARCH_TIMEOUT_OPENALEX_CN", 10, int)
SEARCH_TIMEOUT_S2 = _env("SEARCH_TIMEOUT_S2", 5, int)
SEARCH_TIMEOUT_CORE = _env("SEARCH_TIMEOUT_CORE", 8, int)
SEARCH_TIMEOUT_CROSSREF = _env("SEARCH_TIMEOUT_CROSSREF", 8, int)
SEARCH_TIMEOUT_DOAJ = _env("SEARCH_TIMEOUT_DOAJ", 8, int)
SEARCH_TIMEOUT_PUBMED = _env("SEARCH_TIMEOUT_PUBMED", 8, int)
SEARCH_TIMEOUT_PWC = _env("SEARCH_TIMEOUT_PWC", 8, int)
SEARCH_TIMEOUT_EPMC = _env("SEARCH_TIMEOUT_EPMC", 8, int)
SEARCH_TIMEOUT_GS_APIFY = _env("SEARCH_TIMEOUT_GS_APIFY", 3, int)
SEARCH_TIMEOUT_CNKI = _env("SEARCH_TIMEOUT_CNKI", 2, int)
SEARCH_TIMEOUT_UNPAYWALL = _env("SEARCH_TIMEOUT_UNPAYWALL", 5, int)

# ===================================================================
# Search Limits (per API call)
# ===================================================================
SEARCH_LIMIT_ARXIV = _env("SEARCH_LIMIT_ARXIV", 30, int)
SEARCH_LIMIT_OPENALEX = _env("SEARCH_LIMIT_OPENALEX", 60, int)
SEARCH_LIMIT_OPENALEX_CN = _env("SEARCH_LIMIT_OPENALEX_CN", 60, int)
SEARCH_LIMIT_S2 = _env("SEARCH_LIMIT_S2", 20, int)
SEARCH_LIMIT_CORE = _env("SEARCH_LIMIT_CORE", 15, int)
SEARCH_LIMIT_CROSSREF = _env("SEARCH_LIMIT_CROSSREF", 15, int)
SEARCH_LIMIT_DOAJ = _env("SEARCH_LIMIT_DOAJ", 15, int)
SEARCH_LIMIT_PUBMED = _env("SEARCH_LIMIT_PUBMED", 15, int)
SEARCH_LIMIT_PWC = _env("SEARCH_LIMIT_PWC", 15, int)
SEARCH_LIMIT_EPMC = _env("SEARCH_LIMIT_EPMC", 20, int)
SEARCH_LIMIT_UNPAYWALL = _env("SEARCH_LIMIT_UNPAYWALL", 15, int)

# ===================================================================
# Agent
# ===================================================================
AGENT_MAX_STEPS = _env("AGENT_MAX_STEPS", 10, int)
# Agent 单轮输出上限: 1000 → 4096。Agent 产出是"交付物"(综述/章节/审稿),
# 1000 token 会截断长回答 → 体验差。设宽以容纳完整交付物; 代价是单轮延迟/成本上升。
# 更健壮的方案是检测 finish_reason=="length" 后续写 (未来可做)。
AGENT_MAX_TOKENS = _env("AGENT_MAX_TOKENS", 4096, int)
AGENT_TEMPERATURE = _env("AGENT_TEMPERATURE", 0.3, float)
# Agent 上下文窗口 (context_engine): DeepSeek v4-pro 实际 1M 上下文。
# 论文问答高精度场景: 窗口越大模型看到的证据越多, 幻觉越少 → 用满模型真实窗口。
# reserve 是给系统 prompt + 新输出预留的空间。
AGENT_CONTEXT_MAX_TOKENS = _env("AGENT_CONTEXT_MAX_TOKENS", 1_000_000, int)
AGENT_CONTEXT_RESERVE_TOKENS = _env("AGENT_CONTEXT_RESERVE_TOKENS", 8192, int)
# Agent 反思重试闭环 (loop-engineering 第一环): 直答后轻量自批判,
# 答案不完整 → 带批判意见重试一轮 (有界, 最多 1 次)。false=关闭 (省一次 LLM 调用)。
AGENT_REFLEXION_ENABLED = _env_bool("AGENT_REFLEXION_ENABLED", True)
# Agent 可选 plan 阶段 (loop-engineering 规划层): 长任务先让 LLM 拆 3-6 步计划
# 再执行, 防纯 ReAct 长任务 (整篇论文) 跑偏。仅对长度 ≥ 阈值的任务触发 (短问答不规划省延迟)。
AGENT_PLAN_ENABLED = _env_bool("AGENT_PLAN_ENABLED", True)
AGENT_PLAN_MIN_CHARS = _env("AGENT_PLAN_MIN_CHARS", 60, int)

# Hook BEFORE_EMIT 敏感词表 (空列表 = 不拦截)。默认仅保留高危命令类,
# 不含 "execute arbitrary code" 等学术安全论文常见词组 (避免误杀)。
HOOK_SENSITIVE_WORDS = _env_list("HOOK_SENSITIVE_WORDS", ["rm -rf /", "DROP TABLE"])
# 工具安全: 非白名单域名是否直接阻止 (True = 严格模式; 默认 False 仅告警,
# 避免阻断出版商等非白名单合法下载源)
TOOL_SECURITY_BLOCK_NON_WHITELIST = _env_bool(
    "TOOL_SECURITY_BLOCK_NON_WHITELIST", False
)

# ===================================================================
# Upload / Download 资源限制 (DeepSeek 审查 #14/#15)
# ===================================================================
# 单 PDF 上传大小上限 (字节)
MAX_UPLOAD_BYTES = _env("MAX_UPLOAD_BYTES", 50 * 1024 * 1024, int)
# /download_bulk 并发下载数 (信号量)
DOWNLOAD_CONCURRENCY = _env("DOWNLOAD_CONCURRENCY", 3, int)

# ===================================================================
# Conversation Memory
# ===================================================================
MEMORY_MAX_RECENT = _env(
    "MEMORY_MAX_RECENT", 5, int
)  # max recent turns kept in full text
MEMORY_HISTORY_TURNS = _env(
    "MEMORY_HISTORY_TURNS", 8, int
)  # max history turns sent to LLM
MEMORY_KG_HISTORY_TURNS = _env(
    "MEMORY_KG_HISTORY_TURNS", 4, int
)  # max history turns for KB mode

# ===================================================================
# Rate Limiting
# ===================================================================
RATE_LIMIT_REQUESTS_PER_SEC = _env("RATE_LIMIT_REQUESTS_PER_SEC", 30, int)

# ===================================================================
# 按量付费配额 (付费闭环: 免费额度体验 → 用量追踪 → 超额充值)
# ===================================================================
QUOTA_ENABLED = _env_bool("QUOTA_ENABLED", True)
QUOTA_FREE_TOKENS = _env("QUOTA_FREE_TOKENS", 100_000, int)  # 免费额度 (LLM token 数)

# Stripe 支付 (真实收款: 充值配额)
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PER_1000_TOKENS = _env(
    "STRIPE_PRICE_PER_1000_TOKENS", 0.05, float
)  # $ / 千 token

# ===================================================================
# CORS
# ===================================================================
CORS_ORIGINS = _env_list(
    "CORS_ORIGINS", ["http://localhost:8001", "http://127.0.0.1:8001"]
)

# ===================================================================
# Frontend Model Options
# ===================================================================
AVAILABLE_LLM_MODELS = _env_list(
    "AVAILABLE_LLM_MODELS", ["deepseek-v4-pro", "deepseek-v4-flash"]
)
AVAILABLE_VISION_MODELS = _env_list(
    "AVAILABLE_VISION_MODELS", ["qwen-vl-max", "qwen-vl-plus", "glm-4v"]
)

# ===================================================================
# RAG Pipeline Parameters
# ===================================================================
# Query rewrite: skip expansion for short queries
REWRITE_MIN_LENGTH = _env("REWRITE_MIN_LENGTH", 15, int)
# Semantic validation: reject expansion if cosine similarity < this
EXPANSION_SIMILARITY_THRESHOLD = _env("EXPANSION_SIMILARITY_THRESHOLD", 0.75, float)
# Coreference resolution: only try if question < this length
COREF_MAX_LENGTH = _env("COREF_MAX_LENGTH", 20, int)
# Complex query: trigger rerank if intent matches these types
COMPLEX_INTENT_TYPES = _env_list("COMPLEX_INTENT_TYPES", ["对比", "原理", "最新"])

# ===================================================================
# Intent Classification Keywords (5 classes)
# ===================================================================
# 内容型数据: 意图分类关键词表, 代码常驻 (非配置旋钮)
INTENT_KEYWORDS = {
    "定义": ["什么是", "是什么", "定义", "概念", "define", "what is"],
    "对比": ["对比", "区别", "优缺点", "比较", "vs", "compare", "哪个更好", "差异"],
    "实操": ["如何", "怎么", "怎样", "方法", "步骤", "how to", "流程", "实现"],
    "最新": ["最新", "前沿", "趋势", "2024", "2025", "2026", "recent", "latest", "突破", "进展"],
    "原理": ["为什么", "原理", "机制", "原因", "why", "mechanism", "证明"],
}

# Intent complexity scores — 替代二元匹配，为意图路由提供连续权重
# 高值 → 更可能触发重排序、更保守的改写阈值
INTENT_COMPLEXITY_SCORES = {
    "定义": 0.3,  # 简单事实检索，不触发重排
    "实操": 0.5,  # 中等复杂度
    "最新": 0.7,  # 较复杂，触发 light 重排
    "原理": 0.8,  # 复杂，需要深度推理
    "对比": 0.9,  # 最复杂，触发 full 重排
    "事实": 0.2,  # default fallback
}

# ===================================================================
# Intent Semantic Layer (三层漏斗第 2 层: 嵌入语义分类)
# ===================================================================
# 规则层未命中时, 用嵌入语义相似度兜底分类意图/技能 (见 intent_classifier.py)。
# 判定标准 (bge 系绝对相似度挤在 ~0.7-0.9 窄带, 单阈值不可靠):
#   INTENT_SEMANTIC_MARGIN — 最优 vs 次优 余弦间距 ≥ 此值才采信
#   INTENT_SEMANTIC_FLOOR  — 且最优绝对分数 ≥ 此值
# 不满足 → 返回 None, 交给大模型+工具层 (pi_agent Function Calling) 兜底。
# MARGIN 越大越保守 (更多请求交给 LLM), 越小越激进 (语义层接管更多常规意图)。
INTENT_SEMANTIC_ENABLED = _env_bool("INTENT_SEMANTIC_ENABLED", True)
INTENT_SEMANTIC_MARGIN = _env("INTENT_SEMANTIC_MARGIN", 0.06, float)
INTENT_SEMANTIC_FLOOR = _env("INTENT_SEMANTIC_FLOOR", 0.60, float)

# ===================================================================
# Intelligent Rerank Trigger (替代 hardcoded COMPLEX_INTENT_TYPES)
# ===================================================================
# 重排序触发条件: 意图复杂度 >= 此阈值 → 触发
RERANK_INTENT_THRESHOLD = _env("RERANK_INTENT_THRESHOLD", 0.6, float)
# 检索质量触发: 前3篇向量的最大余弦相似度均低于此值 → 即使意图简单也触发重排
RERANK_QUALITY_TRIGGER = _env("RERANK_QUALITY_TRIGGER", 0.45, float)
# 结果多样性阈值: 检索到的 chunk 之间的余弦相似度均值 > 此值 → 可能重复，触发重排
RERANK_DOCUMENT_DIVERSITY = _env("RERANK_DOCUMENT_DIVERSITY", 0.85, float)
# Light 模式: 意图在 [0.6, 0.8) → 只做 top-15 精排，不回源
# Full 模式: 意图 >= 0.8 → top-15 精排 + 回源扩展
RERANK_LIGHT_INTENT_MAX = _env("RERANK_LIGHT_INTENT_MAX", 0.79, float)

# ===================================================================
# Adaptive Retrieval Parameters
# ===================================================================
# _compute_adaptive_weight coefficients
ADAPTIVE_SHORT_QUERY_BOOST = _env(
    "ADAPTIVE_SHORT_QUERY_BOOST", 0.2, float
)  # <15 chars → BM25 boost
ADAPTIVE_ACRONYM_BOOST = _env(
    "ADAPTIVE_ACRONYM_BOOST", 0.15, float
)  # acronyms → BM25 boost
ADAPTIVE_QUESTION_PENALTY = _env(
    "ADAPTIVE_QUESTION_PENALTY", -0.15, float
)  # question-form → BM25 penalty
ADAPTIVE_LONG_QUERY_PENALTY = _env(
    "ADAPTIVE_LONG_QUERY_PENALTY", -0.1, float
)  # >40 chars → BM25 penalty
ADAPTIVE_SHORT_THRESHOLD = _env("ADAPTIVE_SHORT_THRESHOLD", 15, int)
ADAPTIVE_LONG_THRESHOLD = _env("ADAPTIVE_LONG_THRESHOLD", 40, int)
ADAPTIVE_DEFAULT_ALPHA = _env("ADAPTIVE_DEFAULT_ALPHA", 0.5, float)  # balanced default
ADAPTIVE_ALPHA_MIN = _env("ADAPTIVE_ALPHA_MIN", 0.05, float)
ADAPTIVE_ALPHA_MAX = _env("ADAPTIVE_ALPHA_MAX", 0.95, float)

# ===================================================================
# Image Filter — 多阶段规则预筛 (在调用多模态模型前过滤无用图)
# ===================================================================
IMAGE_MIN_PIXELS = _env("IMAGE_MIN_PIXELS", 100, int)
IMAGE_MAX_ASPECT_RATIO = _env("IMAGE_MAX_ASPECT_RATIO", 6.0, float)
IMAGE_MIN_COLORS = _env("IMAGE_MIN_COLORS", 5, int)

# Stage 1: 尺寸+比例 (快速淘汰, ~90%噪声)
IMAGE_FILTER_STAGE1_ENABLED = _env_bool("IMAGE_FILTER_STAGE1_ENABLED", True)
# Stage 2: 位置过滤 — 首页/末页跳过, 上下边距30%内跳过
IMAGE_FILTER_SKIP_FIRST_PAGE = _env_bool("IMAGE_FILTER_SKIP_FIRST_PAGE", True)
IMAGE_FILTER_SKIP_LAST_PAGE = _env_bool("IMAGE_FILTER_SKIP_LAST_PAGE", True)
IMAGE_FILTER_EDGE_MARGIN_RATIO = _env("IMAGE_FILTER_EDGE_MARGIN_RATIO", 0.30, float)
# Stage 3: 颜色复杂度 — 单调色 (logo/图标/分割线) 跳过
IMAGE_FILTER_MIN_COLORS_MODE = _env(
    "IMAGE_FILTER_MIN_COLORS_MODE", "quantize"
)  # "quantize" | "histogram"
IMAGE_FILTER_MAX_WHITESPACE_RATIO = _env(
    "IMAGE_FILTER_MAX_WHITESPACE_RATIO", 0.85, float
)
# Stage 4: 去重 (同一论文内相似图只保留一张)
IMAGE_FILTER_DEDUPLICATE = _env_bool("IMAGE_FILTER_DEDUPLICATE", True)
IMAGE_DEDUP_SIMILARITY_THRESHOLD = _env("IMAGE_DEDUP_SIMILARITY_THRESHOLD", 0.92, float)

# ===================================================================
# Spam/Quality Filter Keywords (内容型数据, 代码常驻)
# ===================================================================
SPAM_KEYWORDS = [
    "征稿",
    "会议通知",
    "征文",
    "订阅",
    "广告",
    "约稿",
    "稿约",
    "投稿须知",
    "撤稿",
    "抄袭",
    "学术不端",
    "书评",
    "读者来信",
    "简讯",
    "新闻",
]

# Source credibility weights (higher = more trustworthy)
SOURCE_WEIGHTS = {
    "Semantic Scholar": 10,
    "OpenAlex": 8,
    "arXiv": 7,
    "baidu_xueshu": 3,
    "nssd": 5,
}

# ===================================================================
# Prompt Injection Detection Patterns (内容型数据, 代码常驻)
# ===================================================================
INJECTION_PATTERNS = [
    r"忽略.*指令",
    r"ignore.*instruction",
    r"system\s*:",
    r"你是一个",
    r"you are a",
    r"忘记.*规则",
    r"forget.*rule",
    r"切换角色",
    r"switch.*role",
    r"<<SYS>>",
    r"\[INST\]",
    r"\[SYSTEM\]",
    r"DAN\s*:",
    r"jailbreak",
]

# ===================================================================
# CN→EN Translation — 智能化 (内容型数据, 代码常驻)
# ===================================================================
# LLM 翻译缓存: 已翻译过的查询存本地 JSON, 避免重复调用 LLM
CN_EN_TERMS = {
    "计算机视觉": "computer vision",
    "机器学习": "machine learning",
    "深度学习": "deep learning",
    "自然语言处理": "natural language processing",
    "强化学习": "reinforcement learning",
    "图像识别": "image recognition",
    "目标检测": "object detection",
    "语义分割": "semantic segmentation",
    "人脸识别": "face recognition",
    "知识图谱": "knowledge graph",
    "迁移学习": "transfer learning",
    "联邦学习": "federated learning",
    "图神经网络": "graph neural network",
    "自动驾驶": "autonomous driving",
    "人工智能": "artificial intelligence",
    "大语言模型": "large language model",
    "扩散模型": "diffusion model",
    "多模态": "multimodal",
    "神经网络": "neural network",
    "数据挖掘": "data mining",
    "机器人": "robot",
    "视觉": "vision",
    "图像": "image",
    "视频": "video",
    "检测": "detection",
    "分割": "segmentation",
    "识别": "recognition",
    "分类": "classification",
}

# 翻译缓存: 本地 JSON 文件, 命中即用, 无需 LLM 调用
TRANSLATION_CACHE_ENABLED = _env_bool("TRANSLATION_CACHE_ENABLED", True)
TRANSLATION_CACHE_FILE = _env("TRANSLATION_CACHE_FILE", "data/translation_cache.json")
TRANSLATION_CACHE_MAX_ENTRIES = _env("TRANSLATION_CACHE_MAX_ENTRIES", 500, int)

# jieba 分词: 中文输入自动分词, 组合为多词英文查询提升覆盖率
USE_JIEBA_TOKENIZATION = _env_bool("USE_JIEBA_TOKENIZATION", True)
# 分词后取前 N 个关键词 (按 TF-IDF 权重排序, 若 jieba 不可用则取前 N 个按长度)
CN_TOKENIZATION_TOP_K = _env("CN_TOKENIZATION_TOP_K", 5, int)

# ===================================================================
# Query Expansion — 自适应语义校验阈值
# ===================================================================
# 基础阈值 (原 EXPANSION_SIMILARITY_THRESHOLD)
EXPANSION_SIMILARITY_THRESHOLD_BASE = _env(
    "EXPANSION_SIMILARITY_THRESHOLD_BASE", 0.75, float
)
# 自适应调整: 短查询 (<10字) → 更严格 (0.85), 长查询 (>50字) → 更宽松 (0.60)
EXPANSION_SIMILARITY_ADAPTIVE = {
    "short": {"threshold": 0.85, "max_len": 10},
    "long": {"threshold": 0.60, "min_len": 50},
}
