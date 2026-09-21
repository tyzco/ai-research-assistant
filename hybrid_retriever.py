"""自适应混合检索器 + 两阶段级联重排序。

核心创新：
1. 自适应 RRF 融合：根据问题特征（长度、是否含英文术语、是否短查询）
   动态调整向量检索与 BM25 的融合权重，而非固定 50:50。
2. 两阶段级联排序：Stage1 轻量 MiniLM 快速过滤 → Stage2 bge-reranker 精排。

使用：
    retriever = HybridRetriever("papers_topic123")
    results = retriever.search("什么是ArcFace算法？", top_k=5)
"""

import asyncio
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
                        RERANK_STAGE1_MODEL, local_files_only=True
                    )
        return AdaptiveHybridRetriever._stage1_model

    def _get_stage2(self):
        if AdaptiveHybridRetriever._stage2_model is None:
            with AdaptiveHybridRetriever._model_lock:
                if AdaptiveHybridRetriever._stage2_model is None:
                    AdaptiveHybridRetriever._stage2_model = CrossEncoder(
                        RERANK_MODEL, max_length=512, local_files_only=True
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

        stage1_keep = stage1_k or max(top_k * 5, 25)
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

        # 质量水位管控: 高水位/超载时跳过 stage2 CrossEncoder 精排 (最贵的一环), 直接返回 stage1 结果
        if getattr(self, "_load_level", "low") in ("high", "overload"):
            return stage1_chunks[:top_k]

        # Stage 2: CrossEncoder 精排
        try:
            reranker = self._get_stage2()
            pairs = [[query, c.get("text", "")[:512]] for c in stage1_chunks]
            scores2 = reranker.predict(pairs, convert_to_numpy=True)
            from config import RERANK_MIN_SCORE
            if RERANK_MIN_SCORE > 0:
                valid = [(i, sc) for i, sc in enumerate(scores2) if sc >= RERANK_MIN_SCORE]
                if valid:
                    valid.sort(key=lambda x: x[1], reverse=True)
                    return [stage1_chunks[i] for i, _ in valid[:top_k]]
            stage2_indices = sorted(
                range(len(scores2)), key=lambda i: scores2[i], reverse=True
            )[:top_k]
            return [stage1_chunks[i] for i in stage2_indices]
        except Exception as e:
            logger.error(f"Stage2 rerank failed: {e}")  # DeepSeek 审查 #5
            return stage1_chunks[:top_k]

    def _query_variants(self, query: str) -> list[str]:
        """生成检索 query 变体 (召回优化): 原 query + 去问句词的核心词变体。
        同步无延迟 (不做 LLM 翻译); 最多 2 变体控制耗时。"""
        q = (query or "").strip()
        if not q:
            return [q]
        ql = q.lower()
        stripped = ql
        for w in [
            "什么是", "为什么", "如何", "怎么", "有哪些", "有什么区别",
            "介绍一下", "帮我", "请问", "总结", "说明", "解释", "分析",
            "what is", "what are", "how to", "how does", "how do",
            "explain", "describe", "why do", "why is", "please", "list",
        ]:
            stripped = stripped.replace(w, "")
        stripped = stripped.strip(" ？?。.，,：:；;！!()（）")
        if (
            stripped
            and stripped != ql
            and len(stripped) >= 2
            and len(stripped) <= len(ql)
        ):
            return [q, stripped]
        return [q]

    def _fuse_variants(
        self, variants: list[str], query: str, top_k: int, filter_expr: str | None
    ) -> list[dict]:
        """多变体融合: 每变体独立召回 → 加权 RRF 聚合 → 级联重排 (同步)。"""
        if len(variants) <= 1:
            vec_res = self._vector_recall(query, top_k * 6, filter_expr)
            bm25_res = self._bm25_recall(query, top_k * 6)
            if not bm25_res:
                return self._cascade_rerank(query, vec_res, top_k)
            fused = self._adaptive_rrf(vec_res, bm25_res, query, top_k=top_k * 3)
            return self._cascade_rerank(query, fused, top_k)

        agg: dict = {}
        agg_map: dict = {}
        for vq in variants:
            v_alpha = self._compute_adaptive_weight(vq)
            v_vec = self._vector_recall(vq, top_k * 3, filter_expr)
            v_bm25 = self._bm25_recall(vq, top_k * 3)
            for rank, r in enumerate(v_vec):
                cid = r.get("chunk_id", f"v{rank}")
                agg[cid] = agg.get(cid, 0.0) + (1 - v_alpha) / (60 + rank)
                agg_map.setdefault(cid, r)
            for rank, r in enumerate(v_bm25):
                cid = r.get("chunk_id", f"b{rank}")
                agg[cid] = agg.get(cid, 0.0) + v_alpha / (60 + rank)
                agg_map.setdefault(cid, r)
        merged = [agg_map[cid] for cid in sorted(agg, key=agg.get, reverse=True)][
            : top_k * 3
        ]
        return self._cascade_rerank(query, merged, top_k)

    async def _query_variants_async(self, query: str) -> list[str]:
        """生成检索变体 (异步): 原 query + 去问句词 + CN→EN 缓存翻译。"""
        q = (query or "").strip()
        if not q:
            return [q]
        variants = [q]
        stripped = self._query_variants(q)
        if len(stripped) > 1 and stripped[1] not in variants:
            variants.append(stripped[1])
        if any(19968 <= ord(c) <= 40869 for c in q):
            try:
                from search import _translate_cn_query

                # 翻译最多等 0.8s (缓存未命中时) — 防止慢翻译阻塞检索主路径
                en = await asyncio.wait_for(_translate_cn_query(q), timeout=0.8)
                if en:
                    for term in [t.strip() for t in en.split(",") if t.strip()][:2]:
                        if term and term.lower() not in [v.lower() for v in variants]:
                            variants.append(term)
            except (Exception, asyncio.TimeoutError):
                pass
        # HyDE: 生成假设答案段落作检索变体 (默认关; 与速度优先原则冲突, 需显式开启)
        hyde = await self._hyde_variant_async(q)
        if hyde:
            variants.append(hyde)
        return variants[:4]

    async def _hyde_variant_async(self, query: str) -> str:
        """HyDE (Hypothetical Document Embeddings): 让 LLM 写一段"假设答案"作检索变体。

        原理: 问句与答案段落的嵌入分布有 gap; 用假设答案去检索能跨越该 gap (尤其跨语种/弱召回)。
        代价: +1 次 LLM 调用 (~2-5s)。故默认关 (HYDE_ENABLED), 且限长+限时。
        """
        from config import HYDE_ENABLED, HYDE_MAX_TOKENS, HYDE_TIMEOUT

        if not HYDE_ENABLED or len(query) < 6:
            return ""
        try:
            from openai import AsyncOpenAI

            from config import (
                DEEPSEEK_API_KEY,
                DEEPSEEK_BASE_URL,
                DEEPSEEK_MODEL,
            )

            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    temperature=0.3,
                    max_tokens=HYDE_MAX_TOKENS,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "写一段学术论文中可能出现的、直接回答下面问题的段落"
                                "(假设性内容, 用领域术语, 60-100字, 只输出段落不解释):\n" + query[:200]
                            ),
                        }
                    ],
                    extra_body={"thinking": {"type": "disabled"}},
                    timeout=HYDE_TIMEOUT,
                ),
                timeout=HYDE_TIMEOUT + 1,
            )
            txt = (resp.choices[0].message.content or "").strip()
            return txt[:300] if len(txt) >= 10 else ""
        except (Exception, asyncio.TimeoutError):
            return ""

    # ---- 主入口 ----
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> tuple[list[dict], float]:
        """返回 (检索结果列表, BM25权重alpha)。异步 (支持 CN→EN 翻译变体)。

        alpha 值表示本次检索中 BM25 的贡献比例（0=纯向量, 1=纯关键词）。
        同步调用方请用 search_sync()。
        """
        alpha = self._compute_adaptive_weight(query)
        # 质量水位管控 (load-aware degradation): 高水位跳查询变体(省 LLM 翻译/HyDE + 多变体召回),
        # 低水位走全流程保质量。负载回落自动升档。
        try:
            from load_gate import load_level

            _lvl = load_level()
        except Exception:
            _lvl = "low"
        if _lvl == "overload":
            merged = self._fuse_variants([query], query, top_k, filter_expr)
            return merged, alpha
        if _lvl == "high":
            variants = self._query_variants(query)  # 仅同步去问句词变体, 跳异步翻译/HyDE
        else:
            variants = await self._query_variants_async(query)
        self._load_level = _lvl  # 传给 _cascade_rerank 决定是否跳 stage2
        merged = self._fuse_variants(variants, query, top_k, filter_expr)
        return merged, alpha

    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> tuple[list[dict], float]:
        """同步版 search (给 sync main / 测试用)。"""
        import asyncio

        return asyncio.run(self.search(query, top_k, filter_expr))

        # 级联重排
        return self._cascade_rerank(query, fused, top_k), alpha
