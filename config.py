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
