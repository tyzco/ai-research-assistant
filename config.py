"""配置管理：环境变量 + 默认值。"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- LLM (DeepSeek) ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---- Embedding (本地模型) ----
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "512"))

# ---- Vision Model (多模态图片理解，默认关闭) ----
ENABLE_VISION = os.getenv("ENABLE_VISION", "false").lower() == "true"
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ---- Academic Search ----
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")  # https://console.apify.com 免费注册
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "50"))

# ---- Storage Paths ----
PAPER_CACHE_DIR = PROJECT_ROOT / os.getenv("PAPER_CACHE_DIR", "paper_cache")
LANCEDB_DIR = PROJECT_ROOT / os.getenv("LANCEDB_DIR", "data/lancedb_data")
IMAGE_DIR = PROJECT_ROOT / os.getenv("IMAGE_DIR", "data/images")

# ---- Limits ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# ---- Model Options (前端可切换) ----
AVAILABLE_LLM_MODELS = ["deepseek-chat", "deepseek-reasoner"]
AVAILABLE_VISION_MODELS = ["qwen-vl-max", "qwen-vl-plus", "glm-4v"]
