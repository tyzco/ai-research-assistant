"""配置管理：所有可调参数集中于此。其他模块通过 from config import XXX 引用。"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ===================================================================
# LLM (DeepSeek)
# ===================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# ===================================================================
# Embedding (本地模型)
# ===================================================================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "512"))

# ===================================================================
# Rerank 模型 (延迟加载)
# ===================================================================
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_STAGE1_MODEL = os.getenv("RERANK_STAGE1_MODEL", "all-MiniLM-L6-v2")

# ===================================================================
# Vision Model (默认关闭)
# ===================================================================
ENABLE_VISION = os.getenv("ENABLE_VISION", "false").lower() == "true"
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ===================================================================
# Academic Search
# ===================================================================
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "50"))

# ===================================================================
# Storage Paths
# ===================================================================
PAPER_CACHE_DIR = PROJECT_ROOT / os.getenv("PAPER_CACHE_DIR", "paper_cache")
LANCEDB_DIR = PROJECT_ROOT / os.getenv("LANCEDB_DIR", "data/lancedb_data")
IMAGE_DIR = PROJECT_ROOT / os.getenv("IMAGE_DIR", "data/images")

# ===================================================================
# Retrieval Limits
# ===================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MAX_CONTEXT_LENGTH = int(
    os.getenv("MAX_CONTEXT_LENGTH", "300")
)  # chars per chunk in prompt

# ===================================================================
# LLM Call Parameters (all LLM calls use these defaults)
# ===================================================================
# Generation (answers, reports)
LLM_GEN_TEMPERATURE = 0.3
LLM_GEN_MAX_TOKENS = 500
LLM_GEN_TIMEOUT = 20

# Quick responses (agent/quick, translations)
LLM_QUICK_TEMPERATURE = 0.1
LLM_QUICK_MAX_TOKENS = 400
LLM_QUICK_TIMEOUT = 15

# Strategy generation (search strategy, entity extraction)
LLM_STRATEGY_TEMPERATURE = 0.3
LLM_STRATEGY_MAX_TOKENS = 2000
LLM_STRATEGY_TIMEOUT = 30

# Translation (CN→EN keyword translation)
LLM_TRANSLATE_TEMPERATURE = 0.0
LLM_TRANSLATE_MAX_TOKENS = 50
LLM_TRANSLATE_TIMEOUT = 6

# Coreference resolution (multi-turn)
LLM_COREF_TEMPERATURE = 0.1
LLM_COREF_MAX_TOKENS = 100
LLM_COREF_TIMEOUT = 8

# Query expansion
LLM_EXPAND_TEMPERATURE = 0.3
LLM_EXPAND_MAX_TOKENS = 200
LLM_EXPAND_TIMEOUT = 10

# ===================================================================
# Search Timeouts (per source, seconds)
# ===================================================================
SEARCH_TIMEOUT_ARXIV = 10
SEARCH_TIMEOUT_OPENALEX = 10
SEARCH_TIMEOUT_OPENALEX_CN = 10
SEARCH_TIMEOUT_S2 = 5
SEARCH_TIMEOUT_CORE = 8
SEARCH_TIMEOUT_CROSSREF = 8
SEARCH_TIMEOUT_DOAJ = 8
SEARCH_TIMEOUT_PUBMED = 8
SEARCH_TIMEOUT_PWC = 8
SEARCH_TIMEOUT_EPMC = 8
SEARCH_TIMEOUT_GS_APIFY = 3
SEARCH_TIMEOUT_CNKI = 2
SEARCH_TIMEOUT_UNPAYWALL = 5

# ===================================================================
# Search Limits (per API call)
# ===================================================================
SEARCH_LIMIT_ARXIV = 30
SEARCH_LIMIT_OPENALEX = 60
SEARCH_LIMIT_OPENALEX_CN = 60
SEARCH_LIMIT_S2 = 20
SEARCH_LIMIT_CORE = 15
SEARCH_LIMIT_CROSSREF = 15
SEARCH_LIMIT_DOAJ = 15
SEARCH_LIMIT_PUBMED = 15
SEARCH_LIMIT_PWC = 15
SEARCH_LIMIT_EPMC = 20
SEARCH_LIMIT_UNPAYWALL = 15

# ===================================================================
# Agent
# ===================================================================
AGENT_MAX_STEPS = 10
AGENT_MAX_TOKENS = 1000
AGENT_TEMPERATURE = 0.3

# ===================================================================
# Conversation Memory
# ===================================================================
MEMORY_MAX_RECENT = 5  # max recent turns kept in full text
MEMORY_HISTORY_TURNS = 8  # max history turns sent to LLM
MEMORY_KG_HISTORY_TURNS = 4  # max history turns for KB mode

# ===================================================================
# Rate Limiting
# ===================================================================
RATE_LIMIT_REQUESTS_PER_SEC = 30

# ===================================================================
# Frontend Model Options
# ===================================================================
AVAILABLE_LLM_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]
AVAILABLE_VISION_MODELS = ["qwen-vl-max", "qwen-vl-plus", "glm-4v"]

# ===================================================================
# RAG Pipeline Parameters
# ===================================================================
# Query rewrite: skip expansion for short queries
REWRITE_MIN_LENGTH = 15
# Semantic validation: reject expansion if cosine similarity < this
EXPANSION_SIMILARITY_THRESHOLD = 0.75
# Coreference resolution: only try if question < this length
COREF_MAX_LENGTH = 20
# Complex query: trigger rerank if intent matches these types
COMPLEX_INTENT_TYPES = ["对比", "原理", "最新"]

# ===================================================================
# Intent Classification Keywords (5 classes: definition/compare/practical/recent/principle)
# ===================================================================
INTENT_KEYWORDS = {
    "定义": ["什么是", "是什么", "定义", "概念", "define", "what is"],
    "对比": ["对比", "区别", "优缺点", "比较", "vs", "compare", "哪个更好", "差异"],
    "实操": ["如何", "怎么", "怎样", "方法", "步骤", "how to", "流程", "实现"],
    "最新": ["最新", "前沿", "趋势", "2024", "2025", "2026", "recent", "latest", "突破", "进展"],
    "原理": ["为什么", "原理", "机制", "原因", "why", "mechanism", "证明"],
}

# ===================================================================
# Adaptive Retrieval Parameters
# ===================================================================
# _compute_adaptive_weight coefficients
ADAPTIVE_SHORT_QUERY_BOOST = 0.2       # queries < 15 chars → BM25 boost
ADAPTIVE_ACRONYM_BOOST = 0.15          # queries with acronyms → BM25 boost
ADAPTIVE_QUESTION_PENALTY = -0.15      # question-form queries → BM25 penalty
ADAPTIVE_LONG_QUERY_PENALTY = -0.1     # queries > 40 chars → BM25 penalty
ADAPTIVE_SHORT_THRESHOLD = 15
ADAPTIVE_LONG_THRESHOLD = 40
ADAPTIVE_DEFAULT_ALPHA = 0.5           # balanced default
ADAPTIVE_ALPHA_MIN = 0.05
ADAPTIVE_ALPHA_MAX = 0.95

# ===================================================================
# Image Filter Thresholds
# ===================================================================
IMAGE_MIN_PIXELS = 100                  # min width or height in px
IMAGE_MAX_ASPECT_RATIO = 6.0           # max width/height ratio
IMAGE_MIN_COLORS = 5                    # min unique colors (PIL quantized)

# ===================================================================
# Spam/Quality Filter Keywords
# ===================================================================
SPAM_KEYWORDS = [
    "征稿", "会议通知", "征文", "订阅", "广告", "约稿", "稿约",
    "投稿须知", "撤稿", "抄袭", "学术不端", "书评", "读者来信", "简讯", "新闻",
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
# CN→EN Academic Term Translations (fallback when LLM fails)
# ===================================================================
CN_EN_TERMS = {
    "计算机视觉": "computer vision", "机器学习": "machine learning",
    "深度学习": "deep learning", "自然语言处理": "natural language processing",
    "强化学习": "reinforcement learning", "图像识别": "image recognition",
    "目标检测": "object detection", "语义分割": "semantic segmentation",
    "人脸识别": "face recognition", "知识图谱": "knowledge graph",
    "迁移学习": "transfer learning", "联邦学习": "federated learning",
    "图神经网络": "graph neural network", "自动驾驶": "autonomous driving",
    "人工智能": "artificial intelligence", "大语言模型": "large language model",
    "扩散模型": "diffusion model", "多模态": "multimodal",
    "神经网络": "neural network", "数据挖掘": "data mining",
    "机器人": "robot", "视觉": "vision", "图像": "image", "视频": "video",
    "检测": "detection", "分割": "segmentation", "识别": "recognition",
    "分类": "classification",
}

# ===================================================================
# Prompt Injection Detection Patterns
# ===================================================================
INJECTION_PATTERNS = [
    r"忽略.*指令", r"ignore.*instruction", r"system\s*:",
    r"你是一个", r"you are a", r"忘记.*规则", r"forget.*rule",
    r"切换角色", r"switch.*role", r"<<SYS>>", r"\[INST\]",
    r"\[SYSTEM\]", r"DAN\s*:", r"jailbreak",
]
