"""问答引擎：查询改写校验 + 策略路由 + 混合检索 + 级联重排 + 多轮记忆 + DeepSeek 生成。"""

import logging
import re
import threading
from typing import Any

from config import (
    COMPLEX_INTENT_TYPES,
    COREF_MAX_LENGTH,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EXPANSION_SIMILARITY_ADAPTIVE,
    EXPANSION_SIMILARITY_THRESHOLD_BASE,
    INTENT_COMPLEXITY_SCORES,
    INTENT_KEYWORDS,
    RERANK_INTENT_THRESHOLD,
    RERANK_LONG_QUERY_MIN_LEN,
    REWRITE_MIN_LENGTH,
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


def _get_shared_client():
    """共享 LLM 客户端 (复用连接池)。

    此前本模块 3 处各自 AsyncOpenAI(...) 裸建, 每次调用重新 TCP+TLS 握手 ——
    实测 6 次调用共享客户端比每次新建快 31%。
    """
    try:
        from llm_client import get_client

        return get_client()
    except Exception:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)  # noqa: F811

# ---- Rerank 模型: 进程内共享单例 (见 reranker.py) ----
def _get_reranker():
    """精排模型 (bge-reranker)。委托给 reranker 模块的进程级单例 ——
    此前本模块与 hybrid_retriever 各建一份, 同一模型在内存里存了两次。"""
    from reranker import get_stage2

    return get_stage2()


def _rec_degraded(path: str):
    """记录降级触发 (监控用: 降级是保底, 某路径持续增长=正常路径坏了)。"""
    try:
        from monitor import metrics

        metrics.record_degradation(path)
    except Exception:
        pass


def _rerank(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """CrossEncoder 重排序：从粗召回中精排 top-k。同步版本 (供同步调用)。"""
    if len(chunks) <= top_k:
        return chunks
    try:
        reranker = _get_reranker()
        pairs = [[question, c.get("text", "")[:512]] for c in chunks]
        scores = reranker.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        # 低置信度过滤: 与 hybrid_retriever 用同一套语义 (默认阈值 0 = 不过滤)。
        # 此前本路径完全忽略 RERANK_MIN_SCORE, 两条重排路径行为不一致。
        from config import RERANK_KEEP_BEST_IF_NONE, RERANK_MIN_SCORE

        if RERANK_MIN_SCORE > 0:
            kept = [c for s, c in ranked if s >= RERANK_MIN_SCORE]
            if kept:
                return kept[:top_k]
            return [ranked[0][1]] if RERANK_KEEP_BEST_IF_NONE and ranked else []
        return [c for _, c in ranked[:top_k]]
    except Exception as e:
        logger.warning(f"Rerank failed: {e}, using top-{top_k} by vector score")
        _rec_degraded("rerank")
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
    if not history or len(question) >= COREF_MAX_LENGTH:
        return question
    try:
        client = _get_shared_client()
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
        _rec_degraded("coref")
        return question


async def _expand_query(question: str) -> str:
    """DeepSeek 查询改写。"""
    try:
        client = _get_shared_client()
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
        # 语义校验：改写后与原问题语义一致才采用 (阈值按查询长度自适应, 见 _expansion_threshold)
        if _validate_expansion(question, expanded, _expansion_threshold(question)):
            return expanded
        return question
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")  # DeepSeek 审查 #4
        return question


def _expansion_threshold(question: str) -> float:
    """改写校验阈值按查询长度自适应 (config.EXPANSION_SIMILARITY_ADAPTIVE)。

    短查询信息量少, 改写稍偏就丢原意 → 收紧; 长查询约束强, 可放宽允许多样化改写。
    """
    n = len(question or "")
    short = EXPANSION_SIMILARITY_ADAPTIVE.get("short") or {}
    long_ = EXPANSION_SIMILARITY_ADAPTIVE.get("long") or {}
    if short and n < short.get("max_len", 0):
        return short.get("threshold", EXPANSION_SIMILARITY_THRESHOLD_BASE)
    if long_ and n > long_.get("min_len", 10**9):
        return long_.get("threshold", EXPANSION_SIMILARITY_THRESHOLD_BASE)
    return EXPANSION_SIMILARITY_THRESHOLD_BASE


def _validate_expansion(
    original: str,
    expanded: str,
    threshold: float = EXPANSION_SIMILARITY_THRESHOLD_BASE,
) -> bool:
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
    关键词从 config.INTENT_KEYWORDS 统一读取。

    修复: 原实现返回**首个**命中的标签, 而 INTENT_KEYWORDS 里 "定义" 排第一且含
    高频词 "是什么" —— "CNN的原理是什么" 会被判成简单定义, 既不该跳重排也跳了改写。
    现按命中数取最优, 打平时用 INTENT_COMPLEXITY_SCORES 取更复杂的一方
    (即 config 注释所述 "为意图路由提供连续权重")。
    """
    q = (question or "").lower()
    best_label, best_hits, best_score = None, 0, -1.0
    for label, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if not hits:
            continue
        score = INTENT_COMPLEXITY_SCORES.get(label, 0.0)
        if hits > best_hits or (hits == best_hits and score > best_score):
            best_label, best_hits, best_score = label, hits, score
    if best_label:
        return best_label
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
    """策略路由：判断是否需要 Rerank 重排序。

    连续复杂度分数 (config.INTENT_COMPLEXITY_SCORES) >= RERANK_INTENT_THRESHOLD → 重排;
    分数不足但问题够长 (含多约束/多跳) → 仍然重排。
    修复: 原实现只看 COMPLEX_INTENT_TYPES 标签集合, 已定义的连续分数与阈值从未被引用。
    """
    intent = _classify_intent(question)
    score = INTENT_COMPLEXITY_SCORES.get(intent)
    if score is not None and score >= RERANK_INTENT_THRESHOLD:
        return True
    # 兼容标签集合 (config.COMPLEX_INTENT_TYPES 可独立增删) + 长问句兜底
    return (
        intent in tuple(COMPLEX_INTENT_TYPES)
        or len(question) > RERANK_LONG_QUERY_MIN_LEN
    )


# 复杂意图关键词集：由 config.INTENT_KEYWORDS + COMPLEX_INTENT_TYPES 派生 (DeepSeek 审查 #6)
# 替代原硬编码 ["对比", "区别", "优缺点", "机制"]，新增意图关键词无需改代码
COMPLEX_EXPAND_KEYWORDS = {
    kw for label in COMPLEX_INTENT_TYPES for kw in INTENT_KEYWORDS.get(label, [])
}


# ---- Prompt ----
QA_PROMPT = """你是学术文献精读助手。基于用户上传的论文全文回答提问。

**严格规则**（必须遵守）：
1. 优先使用【全文深度资料】回答，详细有据。
2. 每次引用必须标注：`[论文: 标题, 年份, 定位]`。**定位必须照抄资料里给出的那一个**（形如 `p3` 或 `§2.1 对齐损失`）——原文给什么就写什么，不得改写、不得推算、不得编造页码或章节号。资料标注 `位置未知` 时，就写 `位置未知`。
3. 如果命中了图片描述，在答案中说明"详见附图"，并引用图片。
4. **召回不足时不强行作答**：如果上面的资料**覆盖不到问题所问的主题**（不是"答得不全"，而是"根本没这内容"），
   答案**第一行**必须原样输出 `【库内无相关内容】`，然后说明库内实际是什么主题、缺什么。
   此情况下**不要**用你自己的通用知识把问题答出来冒充库内结论 —— 可以补充说明，但必须紧跟 `（库外通用知识）` 标注，
   且不得给出带具体数字/指标的"结论式"表述。
5. 末尾如有【🔎 补充线索】，列出标题 + 年份 + DOI。
6. **溯源格式**（每个关键论断都要能溯源）：来自【全文深度资料】的论断，紧跟标注 `[论文: 标题, 年份, 定位]`；来自模型通用知识/库外的论断，紧跟标注 `（库外通用知识）`。答案保持完整有据，不要因溯源而删减内容、不要省略已有库内引用。

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
    return_ctx: bool = False,
    on_token=None,
):
    """RAG 问答：指代消解 → 查询改写 → 粗召回 → Rerank 精排 → 生成。

    memory_key: ConversationMemory 的 key。传入 topic_id 使前端 /ask 和 Agent
    工具共享同一份对话记忆。为空时回退到 table_name（Agent 工具场景）。

    on_token: 传入 async 回调则改用**流式**生成, 每个 token 调用一次。
    这样 /ask 与 /ask/stream 共用**同一条**链路 —— 此前 /ask/stream 自己重写了一遍
    简化管线 (一句话 prompt + 3块x2000 + 无定位/无记忆/无消解), 导致提示词规则、
    页码/章节定位、召回不足标记**在用户实际使用的那条路上全部失效。
    """
    from knowledge_base import retrieve_with_images

    client = _get_shared_client()
    mem_key = memory_key or table_name

    # 多轮指代消解：有历史 + 短问题 → 消解"它/这"等指代词
    mem_ctx = ""
    try:
        from conversation_memory import get_memory

        mem = get_memory(mem_key)
        mem_ctx = mem.get_context(question)  # 语义选择: 按当前问题取相关记忆
    except Exception:
        pass
    resolved_question = question
    if mem_ctx and len(question) < COREF_MAX_LENGTH:
        resolved_question = await _resolve_coreference(question, mem_ctx)

    # 查询改写（短问题跳过，节省 2-3s；复杂意图短问题仍改写）
    if len(resolved_question) < REWRITE_MIN_LENGTH and not any(
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
        img_res, _ = await hr.search(
            expanded, top_k=3, filter_expr="is_image = true"
        )  # 修复: 原漏 await → 每请求都抛 "cannot unpack coroutine" → 整块降级, 混合检索从未生效
    except Exception as e:
        logger.warning(f"Hybrid retrieval degraded to retrieve_with_images: {e}")
        _rec_degraded("hybrid_retrieval")
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
        def _locator(r: dict) -> str:
            """定位串: 章节 + 页码都尽量给出。

            DOCX 有显式标题层级但无页码; PDF 两者可以都有 (页码来自 page_offsets,
            章节来自字号启发式)。此前是"有章节就只用章节", 对 PDF 会丢掉页码 ——
            而页码恰恰是读者翻原文最快的定位方式。现改为两者并存: `§3.2 方法 (p7)`。
            """
            sec = (r.get("section") or "").strip()
            pg = r.get("page_number") or 0
            if sec and pg:
                return f"§{sec} (p{pg})"
            if sec:
                return f"§{sec}"
            return f"p{pg}" if pg else "位置未知"

        ft = "\n\n".join(
            f"[{i+1}] {r.get('title','N/A')} ({r.get('year','N/A')}), {_locator(r)}\n{r.get('text','')[:4096]}"
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

        if on_token is not None:
            # 流式分支: 与自己那套 prompt/上下文完全一致, 只是 token 边生成边回调
            stream = await client.chat.completions.create(
                model=model or DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=LLM_GEN_TEMPERATURE,
                max_tokens=LLM_GEN_MAX_TOKENS,
                extra_body={"thinking": {"type": "disabled"}},
                timeout=LLM_GEN_TIMEOUT,
                stream=True,
            )
            _parts: list[str] = []
            async for _ch in stream:
                try:
                    _t = _ch.choices[0].delta.content
                except (IndexError, AttributeError):
                    _t = None
                if _t:
                    _parts.append(_t)
                    await on_token(_t)
            answer = "".join(_parts).strip()
            if not answer:
                answer = "生成失败：LLM服务暂时不可用，请稍后重试。"
        else:
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
            _doi = r.get("doi")
            references.append(
                {
                    "title": r.get("title", "N/A"),
                    "year": r.get("year"),
                    "doi": _doi,
                    # 来源链接: 有 DOI 给 DOI 链接, 否则退回 arXiv / 原文 URL
                    "doi_url": f"https://doi.org/{_doi}" if _doi else None,
                    "source": "fulltext",
                    # 证据片段: 该文献里与本次答案最相关的原句 (供用户当场核对)
                    "evidence_snippet": pick_evidence_snippet(
                        answer, r.get("text", "") or ""
                    ),
                    "locator": _locator(r),
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

    # 程序化数字校验 (零 LLM 调用): 答案里的指标值/年份能否在检索证据中找到。
    # 只在有未核对数字时追加一句**口径明确**的提示 —— 不写成"可能是编的",
    # 因为真实含义是"我们给模型的证据里没有这个数"(也可能是召回没覆盖)。
    _ctx_texts = [
        (c.get("text") or "")[:900]
        for c in list(text_res or []) + list(abs_res or [])
    ]
    # 召回不足的结构化标记: 模型被要求在覆盖不到主题时首行输出此标记。
    # 用固定标记而非再调一次 LLM 判断 —— 后者每题 +1 次调用, 与"保速度"冲突;
    # 而检索分数实测无法区分域内/域外 (域外问题 top1 相似度甚至高于域内), 也不能用。
    _insufficient = answer.lstrip().startswith("【库内无相关内容】")
    numeric = {} if _insufficient else (
        verify_numeric_claims(answer, _ctx_texts) if _ctx_texts else {}
    )
    # 引用定位校验: 只认我们**本次真的给过**的定位串 (§章节 (pN) / pN / 位置未知)。
    # 实测 prompt 明写"不得推算不得编造章节号"仍拦不住 —— 模型把不存在的
    # §2.1 ARCHITECTURE 写进了引用。可程序化核验的就不该只靠模型自律。
    _valid_locators = {_locator(r) for r in (text_res or [])}
    cite = {} if _insufficient else verify_citation_locators(answer, _valid_locators)
    if cite.get("fabricated"):
        _bad = "、".join(cite["fabricated"][:5])
        answer = (
            f"{answer}\n\n---\n**引用定位核对**：{_bad} 不是本次检索资料给出的定位"
            "（资料里定位形如 `§CONVNET CONFIGURATIONS (p2)`）。请以资料给出的定位为准。"
        )
        _rec_degraded("citation_fabricated")
    if numeric.get("unverified"):
        _vals = "、".join(u["value"] for u in numeric["unverified"][:5])
        answer = (
            f"{answer}\n\n---\n**数字核对**：{_vals} 未出现在本次检索到的证据片段中。"
            "可能是原文未召回，也可能是表述有误 —— 引用前请回到原文核对。"
        )
        _rec_degraded("numeric_unverified")
    if _insufficient:
        _rec_degraded("insufficient_recall")

    _resp = AskResponse(
        answer=answer,
        references=references,
        supplement=supplement,
        images=image_paths,
        numeric_check=numeric,
        insufficient=_insufficient,
        citation_check=cite,
    )
    # 评估用: 返回本次答案**实际使用**的上下文 (含父子扩展块), 供 faithfulness 判官对齐 —
    # 避免判官用另一条检索路径的 chunk 判"不被支撑"而低估 faithfulness。
    if return_ctx:
        return _resp, [c.get("text", "")[:900] for c in (text_res or [])]
    return _resp


# 高敏感数字: 指标值/百分比/小数/年份/带量级单位 —— 这些编错了最误导, 也最容易被照抄
_NUM_PATTERNS = [
    ("percent", re.compile(r"\d+(?:\.\d+)?\s*%")),
    ("decimal", re.compile(r"\b\d+\.\d+\b")),
    ("year", re.compile(r"\b(?:19|20)\d{2}\b")),
    ("scaled", re.compile(r"\b\d+(?:\.\d+)?\s*(?:倍|万|亿|[kKmMbB])\b")),
]


def _norm_num(s: str) -> str:
    """归一化数字串用于比对: 去空格, 统一大小写。"""
    return re.sub(r"\s+", "", s or "").lower()


def verify_numeric_claims(answer: str, contexts: list[str]) -> dict:
    """**程序化**校验: 答案里的高敏感数字是否能在检索证据中找到。零 LLM 调用。

    对应"对数字、时间等敏感字段添加程序化校验"。为什么值得做:
      - 数字是幻觉里代价最高的一类 (指标/年份写错, 读者无从察觉)
      - 又是唯一能**确定性**校验的一类 (字符串比对, 不依赖另一个 LLM 的判断)
      - 成本近似 0, 可与生成同步完成

    判定口径 (诚实): 标为 unverified 的含义是「**我们给模型的证据里没有这个数**」,
    不等于「模型编造」—— 也可能是召回没覆盖到原文那一段。所以文案必须区分,
    不能对用户说"这是编的"。这正是它比 LLM 自查更可靠的地方: 不猜, 只陈述事实。
    """
    if not answer:
        return {"checked": 0, "unverified": [], "unverified_ratio": 0.0}
    hay = _norm_num(" ".join(contexts or []))
    # DOI / URL 里的数字是**引用标识符**, 不是需要核对真伪的指标值; 且 fulltext 上下文
    # 本来就不带 DOI 字段 (只有摘要层带), 不排除会造成大量误报。先抹掉再扫。
    answer = re.sub(r"10\.\d{4,9}/[^\s，。；)]+", " ", answer or "")
    answer = re.sub(r"https?://\S+", " ", answer)
    found: list[str] = []
    unverified: list[dict] = []
    seen: set[str] = set()

    for kind, pat in _NUM_PATTERNS:
        for m in pat.finditer(answer):
            raw = m.group(0)
            # 去重键取"数值核心"(如 98.35), 否则 98.35% 与 98.35 会被 percent/decimal
            # 两个模式各报一次, 同一个数在报告里出现两遍
            core = re.search(r"\d+(?:\.\d+)?", raw)
            key = _norm_num(core.group(0)) if core else _norm_num(raw)
            if not key or key in seen:
                continue
            seen.add(key)
            if key in hay:
                found.append(raw)
                continue
            # 年份单独放宽: 上下文里常以 (2023) / 2023. 等形式出现, 纯数字比对已覆盖;
            # 这里只保留"证据里确实没有"的
            start = max(0, m.start() - 40)
            unverified.append(
                {
                    "value": raw,
                    "kind": kind,
                    "context": answer[start : m.end() + 40].replace("\n", " "),
                }
            )

    checked = len(found) + len(unverified)
    return {
        "checked": checked,
        "verified": len(found),
        "unverified": unverified[:10],
        "unverified_ratio": (len(unverified) / checked) if checked else 0.0,
    }


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def pick_evidence_snippet(answer: str, chunk_text: str, max_len: int = 220) -> str:
    """从证据块里挑出与答案最相关的那一句, 作为可核验的**证据片段**。

    为什么要有片段而不是只有出处: "见论文《X》第 3 页" 读者仍要自己翻; 给出原句
    才能当场核对。零 LLM 调用 —— 用字符 bigram 重合度打分, 够用且确定性。
    """
    if not chunk_text:
        return ""
    answer = answer or ""

    def bigrams(s: str) -> set:
        s = re.sub(r"\s+", "", s)
        return {s[i : i + 2] for i in range(max(0, len(s) - 1))}

    a = bigrams(answer)
    if not a:
        return ""
    best, best_score = "", -1.0
    for s in _SENT_SPLIT.split(chunk_text):
        s = (s or "").strip()
        # 门槛提到 20 字: 12 字以下多是标题/页眉/表头 ("VERY DEEP CONVOLUTIONAL NETWORKS"),
        # 它们 bigram 少, 一个词命中就能拿高分, 会把真正的证据句挤掉
        if len(s) < 20:
            continue
        # 标题/页眉过滤: 学术 PDF 的标题行多为全大写英文且无句读, 长度也够,
        # 会被当成"证据句"选出来 ("FOR LARGE-SCALE IMAGE RECOGNITION")。
        # 判据: 无中日韩字符 且 字母中大写占比 > 70% → 视为表头, 跳过。
        if not re.search(r"[一-鿿]", s):
            letters = [c for c in s if c.isalpha()]
            if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
                continue
        g = bigrams(s)
        if not g:
            continue
        # 除以 sqrt(len): 纯 ratio 对短串严重偏置 (短串分母小), 开方后缓解又不至于
        # 反过来只挑最长的句子
        score = len(a & g) / (len(g) ** 0.5)
        if score > best_score:
            best, best_score = s, score
    if not best:
        return ""
    return best[:max_len] + ("…" if len(best) > max_len else "")


_CITE_RE = re.compile(r"[\[［]\s*论文\s*[:：]\s*([^\]］]+)[\]］]")


def verify_citation_locators(answer: str, valid_locators: set[str]) -> dict:
    """**程序化**校验: 答案里每条引用标注的"定位"是否真的来自我们给的资料。

    为什么必须做成代码而非 prompt 约束: 实测 prompt 里写明"定位必须照抄资料里给出的
    那一个, 不得推算、不得编造章节号", 模型**仍然**编了 —— 它把 VGG 论文的
    "§2.1 ARCHITECTURE"/"§4.3 MULTI-CROP EVALUATION" 写进引用, 而我们检出的 12 个
    章节标题里根本没有这两个 (它自己补了章节号和名字)。
    这与最初的"页码编造"是同一类问题: **凡是可以程序化核验的, 就不该只靠模型自律**。

    口径 (诚实): 标为 unverified 的含义是「这个定位不是我们提供的」, 可能是编造,
    也可能是把两个真实定位拼在了一起。所以文案说明"非本次检索给出的定位", 不说"造假"。
    """
    valid = {v.strip() for v in (valid_locators or set()) if v and v.strip()}
    # 允许"§章节 (pN)"形式的整体匹配, 也允许单独 §章节 / pN 命中
    valid_sections = {v for v in valid}
    checked = verified = 0
    bad: list[str] = []
    for m in _CITE_RE.finditer(answer or ""):
        body = m.group(1)
        parts = [x.strip() for x in re.split(r"[,，]", body) if x.strip()]
        if not parts:
            continue
        loc = parts[-1]
        # 最后一段不是定位 (没有 § 也没有 p+数字) → 该引用没写定位, 跳过
        if not (loc.startswith("§") or re.match(r"^(位置未知|p\d+)$", loc)):
            continue
        checked += 1
        if loc in valid_sections or loc == "位置未知":
            verified += 1
        elif loc.startswith("§"):
            # 章节名做了截断/加编号都不算命中; 只认我们给过的原文
            if any(loc[1:].strip() == v[1:].strip() for v in valid_sections if v.startswith("§")):
                verified += 1
            else:
                bad.append(loc)
        else:
            bad.append(loc)
    return {
        "checked": checked,
        "verified": verified,
        "fabricated": bad[:10],
        "fabricated_ratio": (len(bad) / checked) if checked else 0.0,
    }


async def ask_with_context(table_name: str, question: str) -> tuple[str, list[str]]:
    """返回答案 + 检索到的上下文块列表（供 RAGAS 评估用）。"""
    from knowledge_base import retrieve_with_images

    expanded = await _expand_query(question)
    text_res, img_res, abs_res = retrieve_with_images(table_name, expanded, top_k=20)
    text_res = _rerank(question, text_res, top_k=5)
    # 判官须看到答案**实际使用**的上下文 (hybrid + 父子扩展), 而非另一条检索路径的 chunk。
    # 注: 上方的 retrieve_with_images 仅用于兜底/空结果提示, 真正上下文从 ask_question 取回。
    result, contexts = await ask_question(
        table_name, question, use_rerank=True, return_ctx=True
    )
    if not contexts:
        contexts = [c.get("text", "")[:900] for c in (text_res or [])]
    return result.answer, contexts


def _cnki_url(title: str) -> str:
    import urllib.parse

    return f"https://kns.cnki.net/kns8/defaultresult/index?kwd={urllib.parse.quote(title[:50])}"


def _gs_url(title: str) -> str:
    import urllib.parse

    return f"https://scholar.google.com/scholar?q={urllib.parse.quote(title[:80])}"
