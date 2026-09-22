"""问答引擎：查询改写校验 + 策略路由 + 混合检索 + 级联重排 + 多轮记忆 + DeepSeek 生成。"""

import logging
import re
import threading
from typing import Any

from openai import AsyncOpenAI

from config import (
    COMPLEX_INTENT_TYPES,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    INTENT_KEYWORDS,
    LLM_COREF_MAX_TOKENS,
    LLM_COREF_TEMPERATURE,
    LLM_COREF_TIMEOUT,
    LLM_EXPAND_MAX_TOKENS,
    LLM_EXPAND_TEMPERATURE,
    LLM_EXPAND_TIMEOUT,
    LLM_GEN_MAX_TOKENS,
    LLM_GEN_TEMPERATURE,
    LLM_GEN_TIMEOUT,
)
from models import AskResponse
from monitor import safe_call

logger = logging.getLogger(__name__)

# ---- Rerank 模型（延迟加载）----
_reranker = None
_reranker_lock = threading.Lock()  # DeepSeek 审查 #3: 防并发重复加载


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                from config import RERANK_MODEL

                logger.info(f"Loading {RERANK_MODEL}...")
                _reranker = CrossEncoder(
                    RERANK_MODEL, max_length=512, local_files_only=True
                )
    return _reranker


def _rerank(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """CrossEncoder 重排序：从粗召回中精排 top-k。同步版本 (供同步调用)。"""
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


async def _rerank_async(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """异步重排 (DeepSeek 审查 #2: CrossEncoder.predict 同步阻塞事件循环)。
    通过 to_thread 放入线程池执行, 不阻塞 asyncio 主循环。"""
    import asyncio

    if len(chunks) <= top_k:
        return chunks
    return await asyncio.to_thread(_rerank, question, chunks, top_k)


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
            temperature=LLM_COREF_TEMPERATURE,
            max_tokens=LLM_COREF_MAX_TOKENS,
            max_retries=1,
            timeout=LLM_COREF_TIMEOUT,
            extra_body={"thinking": {"type": "disabled"}},  # 推理模型必须禁 thinking, 否则 token 耗在 reasoning 上 content 为空
            source="llm",
        )
        if resp is None:
            return question
        resolved = resp.choices[0].message.content.strip()
        return resolved if len(resolved) >= 5 else question
    except Exception as e:
        logger.warning(f"Coreference resolution failed: {e}")  # DeepSeek 审查 #4
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
            temperature=LLM_EXPAND_TEMPERATURE,
            max_tokens=LLM_EXPAND_MAX_TOKENS,
            max_retries=1,
            timeout=LLM_EXPAND_TIMEOUT,
            extra_body={"thinking": {"type": "disabled"}},  # 同上: 禁 thinking 保证 content 非空
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
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")  # DeepSeek 审查 #4
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


def _classify_intent(question: str) -> str:
    """意图路由：分类查询类型以选择最优检索策略。
    三层漏斗第 1+2 层: 关键词快路径 → 嵌入语义兜底 → 默认事实。
    关键词从 config.INTENT_KEYWORDS 统一读取。"""
    q = (question or "").lower()
    for label, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return label
    # 第 2 层: 规则未命中 → 嵌入语义分类 (承接"意思常规但改了表达"的查询)
    try:
        from intent_classifier import classify_intent

        sem = classify_intent(question)
        if sem:
            return sem[0]
    except Exception:
        pass  # 语义层不可用 → 静默降级为纯规则+LLM
    return "事实"  # default: simple fact retrieval


def _is_complex_query(question: str) -> bool:
    """策略路由：判断是否需要 Rerank 重排序。"""
    intent = _classify_intent(question)
    return intent in tuple(COMPLEX_INTENT_TYPES) or len(question) > 15


# 复杂意图关键词集：由 config.INTENT_KEYWORDS + COMPLEX_INTENT_TYPES 派生 (DeepSeek 审查 #6)
# 替代原硬编码 ["对比", "区别", "优缺点", "机制"]，新增意图关键词无需改代码
COMPLEX_EXPAND_KEYWORDS = {
    kw for label in COMPLEX_INTENT_TYPES for kw in INTENT_KEYWORDS.get(label, [])
}


# ---- Prompt ----
QA_PROMPT = """你是学术文献精读助手。基于用户上传的论文全文回答提问。

**严格规则**（必须遵守）：
1. 优先使用【全文深度资料】回答，详细有据。
2. 每次引用必须标注：`[论文: 标题, 年份, 页码大约 N]`。
3. 如果命中了图片描述，在答案中说明"详见附图"，并引用图片。
4. 如果问题无法完全回答，明确指出不足。
5. 末尾如有【🔎 补充线索】，列出标题 + 年份 + DOI。
6. **溯源格式**（每个关键论断都要能溯源）：来自【全文深度资料】的论断，紧跟标注 `[论文: 标题, 年份, pN]`；来自模型通用知识/库外的论断，紧跟标注 `（库外通用知识）`。答案保持完整有据，不要因溯源而删减内容、不要省略已有库内引用。

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
    table_name: str,
    question: str,
    use_rerank: bool = True,
    model: str = "",
    memory_key: str = "",
    extra_context: list[dict] | None = None,
) -> AskResponse:
    """RAG 问答：指代消解 → 查询改写 → 粗召回 → Rerank 精排 → 生成。

    memory_key: ConversationMemory 的 key。传入 topic_id 使前端 /ask 和 Agent
    工具共享同一份对话记忆。为空时回退到 table_name（Agent 工具场景）。"""
    from knowledge_base import retrieve_with_images

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    mem_key = memory_key or table_name

    # 多轮指代消解：有历史 + 短问题 → 消解"它/这"等指代词
    mem_ctx = ""
    try:
        from conversation_memory import get_memory

        mem = get_memory(mem_key)
        mem_ctx = mem.get_context()
    except Exception:
        pass
    resolved_question = question
    if mem_ctx and len(question) < 20:
        resolved_question = await _resolve_coreference(question, mem_ctx)

    # 查询改写（短问题跳过，节省 2-3s；复杂意图短问题仍改写）
    if len(resolved_question) < 15 and not any(
        kw in resolved_question for kw in COMPLEX_EXPAND_KEYWORDS
    ):
        expanded = resolved_question
    else:
        expanded = await _expand_query(resolved_question)

    # 自适应混合检索 + 快速模式（短问题跳过 Rerank 省 1-2s）
    # DeepSeek 审查 #5: _is_complex_query 内部已含 len>15, 外层冗余判断删除
    use_rerank = use_rerank and _is_complex_query(resolved_question)
    # DeepSeek 审查 #7: 预初始化, 避免降级分支抛异常时 UnboundLocalError
    text_res: list[dict] = []
    img_res: list[dict] = []
    abs_res: list[dict] = []
    try:
        from hybrid_retriever import AdaptiveHybridRetriever

        hr = AdaptiveHybridRetriever(table_name)
        text_res, _ = await hr.search(
            expanded, top_k=5, filter_expr="is_fulltext = true AND is_image = false"
        )
        img_res, _ = hr.search(expanded, top_k=3, filter_expr="is_image = true")
    except Exception as e:
        logger.warning(f"Hybrid retrieval degraded to retrieve_with_images: {e}")
        text_res, img_res, abs_res = retrieve_with_images(
            table_name, expanded, top_k=20
        )
        if use_rerank:
            text_res = await _rerank_async(question, text_res, top_k=5)
            img_res = await _rerank_async(question, img_res, top_k=3)

    # 摘要层用简单检索
    if not abs_res:
        _, _, abs_res = retrieve_with_images(table_name, expanded, top_k=5)

    # 无任何检索结果 → 引导建库 (真实体验修复: 原直接调 LLM 返回空答案,
    # 用户"先提问后建库"会看到空白气泡, 且空上下文浪费一次 LLM 调用)
    if not text_res and not img_res and not abs_res:
        return AskResponse(
            answer=(
                "⚠️ 该课题暂无知识库内容，无法检索到相关论文段落。\n\n"
                "请先完成建库再提问：\n"
                "1. 点论文卡片「📥 构建知识库」批量下载 OA 论文，或\n"
                "2. 直接拖拽 PDF 到页面上传\n\n"
                "建库完成后即可基于论文全文回答。"
            )
        )

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
            # DeepSeek 审查 #2: pid 白名单校验, 防止拼接进 where 子句引发语法错误/注入
            if not pid or ci is None or not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
                continue
            # 查同一论文 chunk_index ±1 的块
            neighbors = (
                tbl.search()
                .where(
                    f"paper_id = '{pid}' AND chunk_index >= {max(1, ci-2)} AND chunk_index <= {ci+2} AND is_fulltext = true"
                )
                .limit(6)
                .to_list()
            )
            for n in neighbors:
                if n.get("chunk_id", "") not in expanded_ids:
                    expanded_ids.add(n["chunk_id"])
                    neighbor_texts.append(n)
    except Exception as e:
        logger.warning(f"Neighbor chunk expansion failed: {e}")
    if neighbor_texts:
        text_res = (text_res or []) + neighbor_texts

    # 构建 Prompt
    if extra_context:
        # 生成后接地增强: 补充证据 (论断级查库补出的支撑段落) 追加进上下文
        text_res = (text_res or []) + extra_context
    if text_res:
        ft = "\n\n".join(
            f"[{i+1}] {r.get('title','N/A')} ({r.get('year','N/A')}), p{r.get('page_number',0)}\n{r.get('text','')[:4096]}"
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
        # DeepSeek 审查 #8: 历史放开头, 避免污染末端的回答指令
        if mem_ctx:
            prompt_content = f"【对话历史】\n{mem_ctx}\n\n{prompt_content}"

        resp = await safe_call(
            client.chat.completions.create,
            model=model or DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=LLM_GEN_TEMPERATURE,
            max_tokens=LLM_GEN_MAX_TOKENS,
            # deepseek-v4-pro 推理模型: 禁用推理, 防止复杂 RAG prompt 下推理耗尽 max_tokens 输出空串
            extra_body={"thinking": {"type": "disabled"}},
            max_retries=2,
            timeout=LLM_GEN_TIMEOUT,
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

        get_memory(mem_key).add_turn(question, answer)
    except Exception:
        pass

    # 引用列表 (DeepSeek 审查 #3: 去重键 DOI/paper_id 优先, 避免同名论文误去重)
    def _ref_key(r: dict) -> str:
        return (
            f"doi:{r.get('doi','').lower()}"
            if r.get("doi")
            else (r.get("paper_id") or r.get("title") or "")
        )

    seen = set()
    references: list[dict[str, Any]] = []
    for r in text_res or []:
        key = _ref_key(r)
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
    seen.update(_ref_key(r) for r in (text_res or []))
    for r in abs_res or []:
        key = _ref_key(r)
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
    contexts = [c.get("text", "")[:900] for c in (text_res or [])]
    result = await ask_question(table_name, question, use_rerank=True)
    return result.answer, contexts


def _cnki_url(title: str) -> str:
    import urllib.parse

    return f"https://kns.cnki.net/kns8/defaultresult/index?kwd={urllib.parse.quote(title[:50])}"


def _gs_url(title: str) -> str:
    import urllib.parse

    return f"https://scholar.google.com/scholar?q={urllib.parse.quote(title[:80])}"
