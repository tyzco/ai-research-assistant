"""重排模型单例 — 进程内共享, 避免同一模型被加载多份。

为什么单独抽一个模块:
  此前 qa_engine._get_reranker() 与 hybrid_retriever._get_stage2() 各自
  `CrossEncoder(RERANK_MODEL, ...)`, 同一个 bge-reranker-base 在内存里存了**两份**
  (实测加载分别耗时 8.57s / 1.72s, 各自占一份显存/内存)。
  而 api.py 的启动预热更离谱: `CrossEncoder(RERANK_MODEL, ...)` 创建后**直接丢弃**
  (没赋值给任何地方) —— 既没起到"预热给后续使用"的作用, 又白占一次加载。

  放在独立模块而不是塞进 qa_engine: hybrid_retriever 依赖 qa_engine 是反向依赖
  (qa_engine 才是上层), 两者共用同一个单例需要中立的落点。
"""

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_stage1 = None  # 轻量预筛 (MiniLM)
_stage2 = None  # 精排 (bge-reranker)


def get_stage1():
    """Stage1 轻量预筛模型 (RERANK_STAGE1_MODEL)。线程安全, 进程内唯一。"""
    global _stage1
    if _stage1 is None:
        with _lock:
            if _stage1 is None:
                from sentence_transformers import SentenceTransformer

                from config import RERANK_STAGE1_MODEL

                logger.info(f"Loading stage1 reranker: {RERANK_STAGE1_MODEL}")
                _stage1 = SentenceTransformer(
                    RERANK_STAGE1_MODEL, local_files_only=True
                )
    return _stage1


def get_stage2():
    """Stage2 精排模型 (RERANK_MODEL)。线程安全, 进程内唯一。"""
    global _stage2
    if _stage2 is None:
        with _lock:
            if _stage2 is None:
                from sentence_transformers import CrossEncoder

                from config import RERANK_MODEL

                logger.info(f"Loading stage2 reranker: {RERANK_MODEL}")
                _stage2 = CrossEncoder(
                    RERANK_MODEL, max_length=512, local_files_only=True
                )
    return _stage2


def preload() -> dict:
    """启动预热: 真正把单例建好并**持有**, 而不是建完丢掉。返回加载结果供日志。"""
    out = {}
    for name, fn in (("stage1", get_stage1), ("stage2", get_stage2)):
        try:
            fn()
            out[name] = "ok"
        except Exception as e:  # 预热失败不影响启动, 首次请求时再试
            logger.warning(f"reranker preload {name} failed: {e}")
            out[name] = f"failed: {e}"
    return out
