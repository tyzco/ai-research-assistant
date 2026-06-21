"""问答引擎：查询改写校验 + 策略路由 + 混合检索 + 级联重排 + 多轮记忆 + DeepSeek 生成。"""

import logging
from typing import Any

from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from models import AskResponse
from monitor import safe_call

logger = logging.getLogger(__name__)

# ---- Rerank 模型（延迟加载）----
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading BAAI/bge-reranker-base...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _reranker


def _rerank(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """CrossEncoder 重排序：从粗召回中精排 top-k。"""
    if len(chunks) <= top_k:
        return chunks
    try:
        reranker = _get_reranker()
        pairs = [[question, c.get("text", "")[:512]] for c in chunks]
        scores = reranker.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
    except Exception as e:
        logger.warning(f"Rerank failed: {e}, using top-{top_k} by vector score")
        return chunks[:top_k]


# ---- 查询改写 ----
QUERY_EXPAND_PROMPT = """将以下研究问题改写为用于检索学术论文的详细查询，包含关键术语和同义词。直接返回改写后的查询，不要解释。
问题：{question}
改写："""

COREF_RESOLVE_PROMPT = """根据对话历史，将用户的最新问题改写为一个独立完整的问题（消解指代，如"它"→具体事物）。直接返回改写后的问题，不要解释。

对话历史：
{history}

最新问题：{question}

独立问题："""


async def _resolve_coreference(question: str, history: str) -> str:
    """多轮指代消解：将"它有什么缺点"→"RAG有什么缺点"。仅在短问题+有历史时触发。"""
    if not history or len(question) >= 20:
        return question
    try:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = await safe_call(
            client.chat.completions.create,
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": COREF_RESOLVE_PROMPT.format(
                        history=history[:800], question=question
                    ),
                }
            ],
            temperature=0.1,
            max_tokens=100,
            max_retries=1,
            timeout=8.0,
            source="llm",
        )
        if resp is None:
            return question
        resolved = resp.choices[0].message.content.strip()
        return resolved if len(resolved) >= 5 else question
    except Exception:
        return question


async def _expand_query(question: str) -> str:
    """DeepSeek 查询改写。"""
    try:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = await safe_call(
            client.chat.completions.create,
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": QUERY_EXPAND_PROMPT.format(question=question),
                }
            ],
            temperature=0.3,
            max_tokens=200,
            max_retries=1,
            timeout=10.0,
            source="llm",
        )
        if resp is None:
            return question  # 降级：用原查询
        expanded = resp.choices[0].message.content.strip()
        if len(expanded) < 10:
            return question
        # 语义校验：改写与原问题相似度不低于 0.75，避免引入噪声
        if _validate_expansion(question, expanded):
            return expanded
        return question
    except Exception:
        return question


def _validate_expansion(original: str, expanded: str, threshold: float = 0.75) -> bool:
    """校验改写后的查询是否语义一致。用 bge 嵌入 + 余弦相似度。"""
    import numpy as np

    from knowledge_base import _embed_texts_sync

    emb = _embed_texts_sync([original, expanded])
    a, b = np.array(emb[0]), np.array(emb[1])
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    logger.debug(f"Query expansion similarity: {sim:.3f}")
    return sim >= threshold


def _is_complex_query(question: str) -> bool:
    """策略路由：判断是否需要 Rerank 重排序。"""
    complex_kw = [
        "对比",
        "区别",
        "优缺点",
        "最新",
        "挑战",
        "未来",
        "趋势",
        "vs",
        "compare",
        "深入",
        "详细",
        "机制",
        "原理",
        "为什么",
        "how",
        "why",
    ]
    q = (question or "").lower()
    # 含复杂关键词或长度 > 15 字 → 复杂查询
    if any(kw in q for kw in complex_kw):
        return True
    return len(question) > 15


# ---- Prompt ----
QA_PROMPT = """你是学术文献精读助手。基于用户上传的论文全文回答提问。

**严格规则**（必须遵守）：
1. 优先使用【全文深度资料】回答，详细有据。
2. 每次引用必须标注：`[论文: 标题, 年份, 页码大约 N]`。
3. 如果命中了图片描述，在答案中说明"详见附图"，并引用图片。
4. 如果问题无法完全回答，明确指出不足。
5. 末尾如有【🔎 补充线索】，列出标题 + 年份 + DOI。

---
【全文深度资料（用户上传）】
{fulltext_context}

【图表资料（用户论文中的图片描述）】
{image_context}

【🔎 补充线索（公开摘要，用户库内无全文）】
{abstract_context}
---

用户提问：{question}

回答："""


# ---- 主要问答函数 ----
async def ask_question(
    table_name: str, question: str, use_rerank: bool = True, model: str = ""
) -> AskResponse:
    """RAG 问答：指代消解 → 查询改写 → 粗召回 → Rerank 精排 → 生成。"""
    from knowledge_base import retrieve_with_images

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # 多轮指代消解：有历史 + 短问题 → 消解"它/这"等指代词
    mem_ctx = ""
    try:
        from conversation_memory import get_memory

        mem = get_memory(table_name)
        mem_ctx = mem.get_context()
    except Exception:
        pass
    resolved_question = question
    if mem_ctx and len(question) < 20:
        resolved_question = await _resolve_coreference(question, mem_ctx)

    # 查询改写（短问题跳过，节省 2-3s）
    if len(resolved_question) < 15 and not any(
        kw in resolved_question for kw in ["对比", "区别", "优缺点", "机制"]
    ):
        expanded = resolved_question
    else:
        expanded = await _expand_query(resolved_question)

    # 自适应混合检索 + 快速模式（短问题跳过 Rerank 省 1-2s）
    use_rerank = (
        use_rerank
        and _is_complex_query(resolved_question)
        and len(resolved_question) > 15
    )
    try:
        from hybrid_retriever import AdaptiveHybridRetriever

        hr = AdaptiveHybridRetriever(table_name)
        text_res, _ = hr.search(
            expanded, top_k=5, filter_expr="is_fulltext = true AND is_image = false"
        )
        img_res, _ = hr.search(expanded, top_k=3, filter_expr="is_image = true")
        abs_res = []  # 摘要层用下面的简单检索
    except Exception:
        text_res, img_res, abs_res = retrieve_with_images(
            table_name, expanded, top_k=20
        )
        if use_rerank:
            text_res = _rerank(question, text_res, top_k=5)
            img_res = _rerank(question, img_res, top_k=3)

    # 摘要层用简单检索
    if not abs_res:
        _, _, abs_res = retrieve_with_images(table_name, expanded, top_k=5)

    # 父子分块扩展：检索到 chunk 后，拉取相邻 chunk 提供完整上下文
    neighbor_texts = []
    try:
        import lancedb

        from config import LANCEDB_DIR
        from knowledge_base import _get_table

        db = lancedb.connect(str(LANCEDB_DIR))
        tbl = _get_table(db, table_name)
        expanded_ids = set()
        for r in text_res or []:
            cid = r.get("chunk_id", "")
            if cid in expanded_ids:
                continue
            expanded_ids.add(cid)
            pid = r.get("paper_id", "")
            ci = r.get("chunk_index")
            if not pid or ci is None:
                continue
            # 查同一论文 chunk_index ±1 的块
            neighbors = (
                tbl.search()
                .where(
                    f"paper_id = '{pid}' AND chunk_index >= {max(1, ci-1)} AND chunk_index <= {ci+1} AND is_fulltext = true"
                )
                .limit(6)
                .to_list()
            )
            for n in neighbors:
                if n.get("chunk_id", "") not in expanded_ids:
                    expanded_ids.add(n["chunk_id"])
                    neighbor_texts.append(n)
    except Exception:
        pass  # 静默降级，不影响主流程
    if neighbor_texts:
        text_res = (text_res or []) + neighbor_texts

    # 构建 Prompt
    if text_res:
        ft = "\n\n".join(
            f"[{i+1}] {r.get('title','N/A')} ({r.get('year','N/A')}), p{r.get('page_number',0)}\n{r.get('text','')[:1200]}"
            for i, r in enumerate(text_res)
        )
    else:
        ft = "（暂无用户上传的论文全文）"

    if img_res:
        im = "\n\n".join(
            f"[图{i+1}] {r.get('title','N/A')}, p{r.get('page_number','?')}\n描述: {r.get('text','')[:500]}"
            for i, r in enumerate(img_res)
        )
    else:
        im = "（无图表资料）"

    if abs_res:
        ab = "\n\n".join(
            f"[A{i+1}] {r.get('title','N/A')} ({r.get('year','N/A')})\nDOI: {r.get('doi','N/A')}\n{r.get('text','')[:300]}"
            for i, r in enumerate(abs_res)
        )
    else:
        ab = "（无补充线索）"

    try:
        prompt_content = QA_PROMPT.format(
            fulltext_context=ft,
            image_context=im,
            abstract_context=ab,
            question=question,
        )
        if mem_ctx:
            prompt_content += f"\n\n【对话历史】\n{mem_ctx}"

        resp = await safe_call(
            client.chat.completions.create,
            model=model or DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=0.3,
            max_tokens=800,
            max_retries=2,
            timeout=30.0,
            source="llm",
        )
        if resp is None:
            answer = "生成失败：LLM服务暂时不可用，请稍后重试。"
        else:
            answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        answer = f"生成失败：{e}"

    # 保存多轮记忆
    try:
        from conversation_memory import get_memory

        get_memory(table_name).add_turn(question, answer)
    except Exception:
        pass

    # 引用列表
    seen = set()
    references: list[dict[str, Any]] = []
    for r in text_res or []:
        key = r.get("title", "")
        if key and key not in seen:
            seen.add(key)
            references.append(
                {
                    "title": r.get("title", "N/A"),
                    "year": r.get("year"),
                    "doi": r.get("doi"),
                    "source": "fulltext",
                }
            )

    supplement: list[dict[str, Any]] = []
    seen.update(r.get("title", "") for r in (text_res or []))
    for r in abs_res or []:
        key = r.get("title", "")
        if key and key not in seen:
            seen.add(key)
            doi_val = r.get("doi")
            supplement.append(
                {
                    "title": r.get("title", "N/A"),
                    "year": r.get("year"),
                    "doi": doi_val,
                    "doi_url": f"https://doi.org/{doi_val}" if doi_val else None,
                    "cnki_url": _cnki_url(r.get("title", "")),
                    "google_scholar_url": _gs_url(r.get("title", "")),
                    "abstract_snippet": (r.get("text", "") or "")[:200],
                    "access_guidance": "可通过 DOI 链接访问" if doi_val else "建议通过学校图书馆检索全文",
                }
            )

    image_paths = [
        r.get("image_path", "") for r in (img_res or []) if r.get("image_path")
    ]
    return AskResponse(
        answer=answer, references=references, supplement=supplement, images=image_paths
    )


async def ask_with_context(table_name: str, question: str) -> tuple[str, list[str]]:
    """返回答案 + 检索到的上下文块列表（供 RAGAS 评估用）。"""
    from knowledge_base import retrieve_with_images

    expanded = await _expand_query(question)
    text_res, img_res, abs_res = retrieve_with_images(table_name, expanded, top_k=20)
    text_res = _rerank(question, text_res, top_k=5)
    contexts = [c.get("text", "")[:500] for c in (text_res or [])]
    result = await ask_question(table_name, question, use_rerank=True)
    return result.answer, contexts


def _cnki_url(title: str) -> str:
    import urllib.parse

    return f"https://kns.cnki.net/kns8/defaultresult/index?kwd={urllib.parse.quote(title[:50])}"


def _gs_url(title: str) -> str:
    import urllib.parse

    return f"https://scholar.google.com/scholar?q={urllib.parse.quote(title[:80])}"
