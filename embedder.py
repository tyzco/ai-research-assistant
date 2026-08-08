"""嵌入模型抽象层：支持 bge-small-zh (512d) / bge-m3 (1024d) 热切换。

切换方式：修改 .env 或 config.py 中的 EMBEDDING_MODEL 和 EMBEDDING_DIM。

升级步骤 (bge-small-zh → bge-m3)：
  1. pip install FlagEmbedding
  2. 改 .env: EMBEDDING_MODEL=BAAI/bge-m3  EMBEDDING_DIM=1024
  3. python reindex_all.py  # 重建所有 LanceDB 表的向量（维度不兼容自动处理）
  4. python run_eval.py     # 验证 recall 提升

v5.2 修复 (DeepSeek 对话审查):
  - 统一所有后端 L2 归一化 (原: FlagEmbedding 归一化, ST 未归一化 → 相似度失真)
  - 加载后校验实际维度 vs 配置 (原: 配置不一致静默运行 → LanceDB 维度错乱)
  - threading.Lock 保护模型加载 (原: 并发首次加载可能重复加载模型)
  - aembed_texts 异步接口 (原: 同步 CPU 密集会阻塞事件循环)
"""

import asyncio
import logging
import os
import threading

import numpy as np

# 离线 HF 模型加载 (修复: 模型本地缓存, 但 sentence-transformers 每次加载都会
# 联网检查 huggingface.co 更新; 网络不可达时 5 次退避重试 → 服务重启后
# 首次嵌入/Agent 请求挂起 30-60s+, /health 正常但触发嵌入的端点全部超时)。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

_embed_model = None
_current_model_name = None
_embed_dim_cached = None
_load_lock = threading.Lock()

# 后端判断: 显式配置优先, 否则按模型名启发 (DeepSeek 建议: 避免仅靠命名)
SUPPORTED_BACKENDS = {"sentence_transformers", "flag_embedding"}


def _detect_backend(model_name: str) -> str:
    """确定后端: 显式 EMBEDDING_BACKEND 配置优先, 否则按模型名启发。"""
    try:
        from config import EMBEDDING_BACKEND

        if EMBEDDING_BACKEND and EMBEDDING_BACKEND in SUPPORTED_BACKENDS:
            return EMBEDDING_BACKEND
    except (ImportError, AttributeError):
        pass
    if "m3" in model_name.lower() or "flag" in model_name.lower():
        return "flag_embedding"
    return "sentence_transformers"


def _load_model(model_name: str, dim: int):
    """加载嵌入模型。线程安全, 仅加载一次, 加载后校验维度。"""
    global _embed_model, _current_model_name, _embed_dim_cached

    if _embed_model is not None and _current_model_name == model_name:
        return _embed_model

    with _load_lock:
        # 双检锁: 防止并发首次加载重复加载模型
        if _embed_model is not None and _current_model_name == model_name:
            return _embed_model

        logger.info(f"Loading embedding model: {model_name} (dim={dim})")
        backend = _detect_backend(model_name)

        if backend == "flag_embedding":
            from FlagEmbedding import BGEM3FlagModel

            _embed_model = BGEM3FlagModel(model_name, use_fp16=True)
        else:
            from sentence_transformers import SentenceTransformer

            _embed_model = SentenceTransformer(model_name)

        _current_model_name = model_name

        # 维度校验 (DeepSeek 建议: 配置与实际不符提前报错, 而非静默写错 LanceDB)
        try:
            test_vec = _embed_model.encode(["test"], batch_size=1)
            if isinstance(test_vec, dict):
                actual_dim = test_vec["dense_vecs"].shape[1]
            else:
                actual_dim = test_vec.shape[1]
            _embed_dim_cached = actual_dim
            if actual_dim != dim:
                logger.warning(
                    f"Embedding dim mismatch: config={dim}, model={actual_dim}. "
                    f"LanceDB 索引维度由模型决定, 请同步 config.LOCAL_EMBEDDING_DIM={actual_dim}"
                )
        except Exception as e:
            logger.warning(f"Dim validation failed: {e}")

        return _embed_model


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """统一 L2 归一化 (DeepSeek 建议: 所有后端输出单位向量, 保证余弦相似度可比)。"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量编码文本为向量。自动选择正确后端。统一 L2 归一化。"""
    if not texts:
        return []
    from config import LOCAL_EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL

    model = _load_model(LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIM)
    output = model.encode(texts, batch_size=min(32, len(texts)))

    # FlagEmbedding 返回 dict {'dense_vecs': np.array, 'lexical_weights': ...}
    if isinstance(output, dict):
        vecs = output["dense_vecs"]
    else:
        vecs = output

    # 统一归一化 (修复: 原 ST 分支未归一化 → 与 FlagEmbedding 向量不可比)
    return _normalize(np.asarray(vecs, dtype=np.float64)).tolist()


async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """异步批量编码 (DeepSeek 建议: 避免同步 CPU 密集阻塞事件循环)。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts, texts)


def embed_query(text: str) -> list[float]:
    """编码单条查询。"""
    return embed_texts([text])[0]


# 向后兼容别名（避免全局搜索替换）
def _get_embed_model():
    from config import LOCAL_EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL

    return _load_model(LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIM)


_embed_texts_sync = embed_texts
