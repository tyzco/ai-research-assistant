"""自适应混合检索器 + 两阶段级联重排序。

核心创新：
1. 自适应 RRF 融合：根据问题特征（长度、是否含英文术语、是否短查询）
   动态调整向量检索与 BM25 的融合权重，而非固定 50:50。
2. 两阶段级联排序：Stage1 轻量 MiniLM 快速过滤 → Stage2 bge-reranker 精排。

使用：
    retriever = HybridRetriever("papers_topic123")
    results = retriever.search("什么是ArcFace算法？", top_k=5)
"""

import logging
import re
import threading

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import RERANK_MODEL, RERANK_STAGE1_MODEL
from knowledge_base import _cosine_search, _embed_texts_sync

logger = logging.getLogger(__name__)


class AdaptiveHybridRetriever:
    """自适应多路召回 + RRF 融合 + 两级级联重排。"""

    # 类级别模型缓存——多个 topic 共享，避免重复加载
    _stage1_model = None
    _stage2_model = None
    _model_lock = threading.Lock()  # DeepSeek 审查 #1: 防并发重复加载

    def __init__(self, table_name: str):
        import lancedb

        from config import LANCEDB_DIR

        db = lancedb.connect(str(LANCEDB_DIR))
        self.table = db.open_table(table_name)
        self.table_name = table_name

    def _get_stage1(self):
        # 双检锁: 线程安全加载 (DeepSeek 审查 #1)
        if AdaptiveHybridRetriever._stage1_model is None:
            with AdaptiveHybridRetriever._model_lock:
                if AdaptiveHybridRetriever._stage1_model is None:
                    AdaptiveHybridRetriever._stage1_model = SentenceTransformer(
                        RERANK_STAGE1_MODEL
                    )
        return AdaptiveHybridRetriever._stage1_model

    def _get_stage2(self):
        if AdaptiveHybridRetriever._stage2_model is None:
            with AdaptiveHybridRetriever._model_lock:
                if AdaptiveHybridRetriever._stage2_model is None:
                    AdaptiveHybridRetriever._stage2_model = CrossEncoder(
                        RERANK_MODEL, max_length=512
                    )
        return AdaptiveHybridRetriever._stage2_model

    # ---- 自适应权重推理 ----
    def _compute_adaptive_weight(self, query: str) -> float:
        """根据问题特征，返回 BM25 权重（0-1）。
        返回值越高，BM25 关键词匹配越重要。

        特征：
        - 短查询（<15字）：偏关键词 → BM25 高权重
        - 含英文缩写/公式：偏关键词 → BM25 高权重
        - 长查询/自然语言问句：偏语义 → 向量高权重
        """
        score = 0.5  # 默认均衡

        # 短查询：更像关键词搜索
        if len(query) < 15:
            score += 0.2
        # 含英文术语/缩写（如 "ArcFace", "CNN", "LSTM"）
        if re.search(r"[A-Z]{2,}|[a-z]+[A-Z]", query):
            score += 0.15
        # 问句形式：更像自然语言
        if re.search(r"[?？]|什么是|如何|怎么样|有哪些", query):
            score -= 0.15
        # 长查询：更像语义搜索
        if len(query) > 40:
            score -= 0.1

        return max(0.05, min(0.95, score))

    # ---- 多路召回 ----
    def _vector_recall(
        self, query: str, top_k: int, filter_expr: str | None = None
    ) -> list[dict]:
        query_vec = _embed_texts_sync([query])[0]
        return _cosine_search(self.table, query_vec, top_k, filter_expr)

    def _bm25_recall(self, query: str, top_k: int) -> list[dict]:
        try:
            return self.table.search(query, query_type="fts").limit(top_k).to_list()
        except Exception:
            return []

    # ---- 自适应 RRF 融合 ----
    def _adaptive_rrf(
        self,
        vec_results: list[dict],
        bm25_results: list[dict],
        query: str,
        top_k: int,
    ) -> list[dict]:
        """自适应 RRF：BM25 权重根据问题特征动态调整。"""
        alpha = self._compute_adaptive_weight(query)  # BM25 权重
        beta = 1.0 - alpha  # 向量权重

        def weighted_rrf(rank: int, weight: float, k: int = 60) -> float:
            return weight / (k + rank)

        fused: dict[str, tuple[float, dict]] = {}
        for rank, r in enumerate(vec_results):
            cid = r.get("chunk_id", f"v{rank}")
            fused[cid] = (fused.get(cid, (0, r))[0] + weighted_rrf(rank, beta), r)

        for rank, r in enumerate(bm25_results):
            cid = r.get("chunk_id", f"b{rank}")
            fused[cid] = (fused.get(cid, (0, r))[0] + weighted_rrf(rank, alpha), r)

        sorted_items = sorted(fused.items(), key=lambda x: x[1][0], reverse=True)[
            :top_k
        ]
        return [item[1][1] for item in sorted_items]

    # ---- 两阶段级联排序 ----
    def _cascade_rerank(
        self, query: str, chunks: list[dict], top_k: int = 5, stage1_k: int = None
    ) -> list[dict]:
        """Stage1: MiniLM 快速预筛 → Stage2: CrossEncoder 精排。

        v5.3 (DeepSeek 审查 #2): stage1_k 可配置, 默认 max(top_k*3, 15)。
        #8: 点积前 L2 归一化, 保证余弦相似度。"""
        if len(chunks) <= top_k:
            return chunks

        stage1_keep = stage1_k or max(top_k * 3, 15)
        stage1_keep = min(stage1_keep, len(chunks))

        # Stage 1: 轻量语义相似度预筛
        try:
            stage1 = self._get_stage1()
            texts = [c.get("text", "")[:512] for c in chunks]
            q_emb = stage1.encode([query])[0]
            c_embs = stage1.encode(texts)
            # 归一化后点积 = 余弦相似度 (DeepSeek 审查 #8)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
            c_norm = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-12)
            scores = np.dot(c_norm, q_norm)
            stage1_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:stage1_keep]
            stage1_chunks = [chunks[i] for i in stage1_indices]
        except Exception as e:
            logger.error(f"Stage1 rerank failed: {e}")  # DeepSeek 审查 #5
            return chunks[:top_k]

        # Stage 2: CrossEncoder 精排
        try:
            reranker = self._get_stage2()
            pairs = [[query, c.get("text", "")[:512]] for c in stage1_chunks]
            scores2 = reranker.predict(pairs, convert_to_numpy=True)
            stage2_indices = sorted(
                range(len(scores2)), key=lambda i: scores2[i], reverse=True
            )[:top_k]
            return [stage1_chunks[i] for i in stage2_indices]
        except Exception as e:
            logger.error(f"Stage2 rerank failed: {e}")  # DeepSeek 审查 #5
            return stage1_chunks[:top_k]

    # ---- 主入口 ----
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> tuple[list[dict], float]:
        """返回 (检索结果列表, BM25权重alpha)。

        alpha 值表示本次检索中 BM25 的贡献比例（0=纯向量, 1=纯关键词）。
        调用方可记录此值用于消融实验。
        """
        alpha = self._compute_adaptive_weight(query)

        vec_res = self._vector_recall(query, top_k * 4, filter_expr)
        bm25_res = self._bm25_recall(query, top_k * 4)

        if not bm25_res:
            return self._cascade_rerank(query, vec_res, top_k), alpha

        # 自适应 RRF 融合
        fused = self._adaptive_rrf(vec_res, bm25_res, query, top_k=top_k * 2)

        # 级联重排
        return self._cascade_rerank(query, fused, top_k), alpha
