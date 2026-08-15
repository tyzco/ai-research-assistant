"""检索策略生成 + 论文搜索（OA 链接标注）。"""

import asyncio
import hashlib
import json
import logging
import xml.etree.ElementTree as ET

import httpx
from openai import AsyncOpenAI

from config import (
    APIFY_API_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    SEMANTIC_SCHOLAR_API_KEY,
    SOURCE_WEIGHTS,
    SPAM_KEYWORDS,
)
from models import PaperMeta

logger = logging.getLogger(__name__)

STRATEGY_PROMPT = """你是图书情报学与学术检索专家。用户的研究方向是「{query}」，请按以下三层生成检索策略。

## 第1层：核心关键词（提取+分词）
- 从用户输入中提取 3-5 个核心中文关键词，必须分词（每个 ≤5 字），去掉"基于""研究""方法""分析"等停用词
- 翻译为 3-5 个对应的英文标准术语（1-3 词一组）

## 第2层：领域映射与术语联想
- domain_tags: 该课题属于哪 2-3 个学科方向（中英文），如"计算机视觉/Computer Vision"
- related_terms_cn: 联想 6-8 个领域内的规范中文术语，即使用户没提到。例如用户说"人脸识别"，应联想到"特征提取""卷积神经网络""LFW数据集""ArcFace"等
- related_terms_en: 对应的英文本语

## 第3层：专业检索式
为以下 4 个数据库各生成一条布尔逻辑检索式（使用第1、2层的词组合）：
- 知网: 用 AND/OR 连接，如"主题=A AND (关键词=B OR 关键词=C)"
- 万方: 题名或关键词 + 摘要限定
- Web of Science: TS=(...) AND TI=(...) 格式
- arXiv: 英文关键词组合
- 提示：检索式内的词组用单引号包裹，禁止使用双引号

额外要求：
- top_authors: 3-5 位该领域知名学者（姓名+机构+贡献简述）
- top_institutions: 2-3 个该领域强校/实验室
- search_tips: 2-3 条实用检索建议

返回严格 JSON（不要 markdown），格式：
{{
  "keywords_cn": ["词1","词2"],
  "keywords_en": ["kw1","kw2"],
  "domain_tags": ["计算机视觉/Computer Vision", "生物特征识别/Biometrics"],
  "related_terms_cn": ["术语1","术语2","术语3"],
  "related_terms_en": ["term1","term2","term3"],
  "boolean_queries": [
    {{"database":"知网","query":"主题=...","note":"说明"}},
    {{"database":"万方","query":"...","note":"说明"}},
    {{"database":"Web of Science","query":"TS=(...)","note":"说明"}},
    {{"database":"arXiv","query":"...","note":"说明"}}
  ],
  "recommended_databases": ["库1","库2"],
  "top_authors": [{{"name":"姓名","institution":"机构","reason":"贡献"}}],
  "top_institutions": ["机构1"],
  "search_tips": "建议文本"
}}
仅返回 JSON。"""


# ===== 检索/策略缓存 (速度: 重复查询秒回, 跨进程 JSON 持久化) =====
import time as _t
from pathlib import Path as _Path

_STRATEGY_CACHE: dict[str, tuple[float, dict]] = {}
_SEARCH_CACHE: dict[str, tuple[float, list]] = {}
_CACHE_FILE = _Path(__file__).resolve().parent / "data" / "search_cache.json"
_STRATEGY_TTL = 3600  # 策略缓存 1 小时
_SEARCH_TTL = 600  # 搜索结果缓存 10 分钟
_CACHE_MAX = 100


def _cache_get(cache: dict, key: str, ttl: float):
    item = cache.get(key)
    if item and _t.time() - item[0] < ttl:
        return item[1]
    return None


def _cache_set(cache: dict, key: str, value):
    now = _t.time()
    if len(cache) >= _CACHE_MAX:
        for k in [k for k, v in cache.items() if now - v[0] > _SEARCH_TTL]:
            cache.pop(k, None)
    cache[key] = (now, value)


def _load_caches():
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        _STRATEGY_CACHE.update(
            {k: (v["ts"], v["strategy"]) for k, v in data.get("strategy", {}).items()}
        )
        _SEARCH_CACHE.update(
            {k: (v["ts"], v["papers"]) for k, v in data.get("search", {}).items()}
        )
    except Exception:
        pass


def _save_caches():
    try:
        payload = {
            "strategy": {
                k: {"ts": v[0], "strategy": v[1]}
                for k, v in list(_STRATEGY_CACHE.items())[-30:]
            },
            "search": {
                k: {"ts": v[0], "papers": v[1]}
                for k, v in list(_SEARCH_CACHE.items())[-20:]
            },
        }
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


_load_caches()  # 启动时载入跨进程缓存 (演示重跑/服务重启后仍秒回)


async def generate_search_strategy(query: str) -> dict:
    """检索策略 (缓存包装): 同查询秒回, 跨进程 JSON 持久化。"""
    key = (query or "").strip().lower()
    cached = _cache_get(_STRATEGY_CACHE, key, _STRATEGY_TTL)
    if cached is not None:
        # 缓存里的旧坏策略 (英文关键词混中文) 也做校验过滤
        en = cached.get("keywords_en", [])
        en_clean = [k for k in en if k and not any(19968 <= ord(c) <= 40869 for c in k)]
        if en_clean:
            cached["keywords_en"] = en_clean
        logger.info("Strategy cache hit")
        return cached
    strategy = await _generate_strategy_uncached(query)
    _cache_set(_STRATEGY_CACHE, key, strategy)
    _save_caches()
    return strategy


async def _generate_strategy_uncached(query: str) -> dict:
    from config import (
        LLM_STRATEGY_MAX_TOKENS,
        LLM_STRATEGY_TEMPERATURE,
        LLM_STRATEGY_TIMEOUT,
    )

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def _fallback() -> dict:
        """JSON 解析失败时的降级默认值。"""
        return {
            "keywords_cn": [query],
            "keywords_en": [query],
            "domain_tags": [],
            "related_terms_cn": [],
            "related_terms_en": [],
            "boolean_queries": [],
            "recommended_databases": [],
            "top_authors": [],
            "top_institutions": [],
            "search_tips": f"请手动为「{query}」构建检索式",
        }

    # 解析 LLM 输出: 清理 Markdown 标记 + 尾随逗号 → JSON
    def _parse(text: str) -> dict | None:
        import re

        text = text.replace("```json", "").replace("```", "").strip()
        text = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            parsed = json.loads(text)
            if not (isinstance(parsed, dict) and parsed.get("keywords_cn")):
                return None
            # 硬校验: keywords_en 必须是英文术语 (中文原文混入 → 英文源召回失败)
            # 过滤掉含中文的条目; 若全中文 → 视为策略失败, 触发重试
            en = parsed.get("keywords_en", [])
            en_clean = [
                k for k in en if k and not any(19968 <= ord(c) <= 40869 for c in k)
            ]
            if not en_clean:
                return None
            parsed["keywords_en"] = en_clean
            return parsed
        except json.JSONDecodeError:
            pass
        return None

    # 首次调用
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": STRATEGY_PROMPT.format(query=query)}],
        temperature=LLM_STRATEGY_TEMPERATURE,
        max_tokens=LLM_STRATEGY_MAX_TOKENS,
        timeout=LLM_STRATEGY_TIMEOUT,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    parsed = _parse(text)
    if parsed is not None:
        if not (parsed.get("related_terms_en") or parsed.get("related_terms_cn")):
            logger.warning(f"Strategy missing related_terms for: {query}")
        return parsed

    # 重试一次 (DeepSeek 审查 #2: 降低温度提高结构化输出稳定性)
    logger.warning(
        f"Strategy JSON parse failed, retrying with lower temp: {text[:200]}"
    )
    try:
        retry = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": STRATEGY_PROMPT.format(query=query)
                    + "\n\n硬性要求: 1) keywords_en 必须是纯英文术语, 禁止含中文; "
                    "2) 必须生成 related_terms_cn 和 related_terms_en (各 ≥3 个领域联想词); "
                    "3) 严格输出指定 JSON 结构。",
                }
            ],
            temperature=0.0,
            max_tokens=LLM_STRATEGY_MAX_TOKENS,
            timeout=LLM_STRATEGY_TIMEOUT,
            response_format={"type": "json_object"},
        )
        parsed = _parse(retry.choices[0].message.content.strip())
        if parsed is not None:
            return parsed
    except Exception as e:
        logger.warning(f"Strategy retry failed: {e}")

    return _fallback()


# ---------- 质量过滤 & 语言检测 ----------


def _detect_cn(text: str) -> bool:
    """charCode 汉字检测（与前端 isCN 一致）。"""
    for c in text or "":
        if 19968 <= ord(c) <= 40869:
            return True
    return False


def _infer_source(p: PaperMeta) -> str:
    """从 paper_id 前缀/arxiv_id 推断论文来源，用于 SOURCE_WEIGHTS 查权重。
    已知前缀: core/crossref/doaj/pubmed/epmc/pwc/gs。
    无前缀: arxiv_id 存在 → arXiv; 有 doi → 其他 (Semantic Scholar/OpenAlex 无稳定区分)。
    """
    pid = p.paper_id or ""
    if pid.startswith("core:"):
        return "CORE"
    if pid.startswith("crossref:"):
        return "Crossref"
    if pid.startswith("doaj:"):
        return "DOAJ"
    if pid.startswith("pubmed:"):
        return "PubMed"
    if pid.startswith("epmc:"):
        return "Europe PMC"
    if pid.startswith("pwc:"):
        return "Papers With Code"
    if pid.startswith("gs:") or pid.startswith("gs_apify"):
        return "Google Scholar"
    if p.arxiv_id:
        return "arXiv"
    # Semantic Scholar / OpenAlex 无前缀区分 — 按 doi 有无粗分
    return "OpenAlex" if p.doi else "Semantic Scholar"


def _quality_score(p: PaperMeta, source: str = "") -> float:
    """综合质量评分：来源可信度 + 引用数 + 年份加权。

    年份项权重从 0.1 提到 0.5: 修复「结果总是旧论文」——原来引用数(最高40分)碾压年份,
    导致 2003 高引旧文排在 2023 新文之前。0.5 让近期论文有真实竞争力, 又不淹没经典。
    """
    # 来源权重：显式 source 优先，否则从 paper 推断 (修复硬编码 "other" 问题)
    w = (
        SOURCE_WEIGHTS.get(source)
        if source
        else SOURCE_WEIGHTS.get(_infer_source(p), 2)
    )
    # 引用数信号: 0~1000 → 0~5
    cit = p.citation_count or 0
    cit_signal = min(cit / 1000.0, 5.0)
    return w * 10 + cit_signal * 8 + max(0, (p.year or 2000) - 2000) * 0.5


def _should_drop(p: PaperMeta) -> bool:
    """仅丢弃明显垃圾（广告/撤稿/征稿）。保留其他所有论文。"""
    t = (p.title or "") + (p.abstract or "")
    for w in SPAM_KEYWORDS:
        if w in t:
            return True
    # 标题短于4字且无DOI才丢弃
    if len(p.title or "") < 4 and not p.doi:
        return True
    return False


# ---------- 论文搜索（并行 S2 + arXiv + CNKI 链接）----------


async def _translate_cn_query(query: str) -> str:
    """中文学术语 → 英文检索词 (LLM 翻译 + 本地缓存)。无中文/失败返回空。"""
    if not any(19968 <= ord(c) <= 40869 for c in query):
        return ""
    from config import (
        LLM_TRANSLATE_MAX_TOKENS,
        LLM_TRANSLATE_TEMPERATURE,
        LLM_TRANSLATE_TIMEOUT,
        TRANSLATION_CACHE_ENABLED,
        TRANSLATION_CACHE_FILE,
    )

    # 缓存
    try:
        if TRANSLATION_CACHE_ENABLED:
            from pathlib import Path

            p = Path(TRANSLATION_CACHE_FILE)
            if p.exists():
                cache = json.loads(p.read_text())
                if cache.get(query):
                    return cache[query]
    except Exception:
        pass

    try:
        from openai import AsyncOpenAI

        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=LLM_TRANSLATE_TEMPERATURE,
            max_tokens=LLM_TRANSLATE_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": (f"将以下中文学术术语翻译为英文检索关键词，只返回英文逗号分隔，不要解释：{query}"),
                }
            ],
            timeout=LLM_TRANSLATE_TIMEOUT,
        )
        import re

        result = re.sub(
            r"[^a-zA-Z0-9,\- ]", "", (resp.choices[0].message.content or "")
        )[:100]
        result = result.strip()
        if not result:
            return ""
        try:
            if TRANSLATION_CACHE_ENABLED:
                from pathlib import Path

                p = Path(TRANSLATION_CACHE_FILE)
                cache = json.loads(p.read_text()) if p.exists() else {}
                cache[query] = result
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(cache, ensure_ascii=False))
        except Exception:
            pass
        return result
    except Exception as e:
        logger.warning(f"CN→EN translation failed: {e}")
        return ""


S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "title,authors,year,abstract,openAccessPdf,externalIds,journal,citationCount"
)
ARXIV_URL = "https://export.arxiv.org/api/query"


async def search_papers_for_topic(
    query: str,
    keywords_en: list[str] | None = None,
    keywords_cn: list[str] | None = None,
    related_en: list[str] | None = None,
    related_cn: list[str] | None = None,
) -> list[PaperMeta]:
    """9 源并行搜索 + Query扩展: 每源独立超时, 快源先回。参考 academic-search 的 Query 扩展策略。
    related_en/cn: 领域联想词 (策略第2层), 并入关键词池补覆盖 — 不常见选题用户措辞≠标准术语时,
    联想词进 arXiv/OpenAlex/CORE 等 [:2]/[:3] 源检索。"""
    try:
        from monitor import metrics

        metrics.record_search()  # /metrics 搜索计数 (接线 MetricsTracker.record_search)
    except Exception:
        pass
    # 搜索缓存: 同查询+关键词+联想词重复检索秒回 (跳过策略生成 + 12 源检索)
    _cache_key = (
        ((query or "").strip().lower() + "|")
        + "|".join(sorted(keywords_en or []))
        + "|"
        + "|".join(sorted(keywords_cn or []))
        + "|"
        + "|".join(sorted(related_en or []))
        + "|"
        + "|".join(sorted(related_cn or []))
    )
    _cached = _cache_get(_SEARCH_CACHE, _cache_key, _SEARCH_TTL)
    if _cached is not None:
        logger.info(f"Search cache hit: {len(_cached)} papers")
        return [PaperMeta(**p) for p in _cached]
    if not keywords_en:
        # 策略 LLM 不可用/超时 → 降级为查询词本身, 搜索不因策略失败而中断
        try:
            strategy = await generate_search_strategy(query)
            keywords_en = strategy.get("keywords_en", [])[:2]
            keywords_cn = strategy.get("keywords_cn", [])[:3]
            related_en = related_en or strategy.get("related_terms_en", [])
            related_cn = related_cn or strategy.get("related_terms_cn", [])
        except Exception as e:
            logger.warning(f"Strategy generation failed, using query as keyword: {e}")
            keywords_en = keywords_cn = None
    # 修复: 关键词含中文 → 翻译为英文。不常见选题最常见的失败路径是
    # 策略降级返回中文原文, 英文源(arXiv/S2/Crossref)搜中文=召回失败。
    if keywords_en and any(19968 <= ord(c) <= 40869 for c in "".join(keywords_en)):
        en = await _translate_cn_query(query)
        if en:
            keywords_en = [en]
            logger.info(f"Translated query keywords → EN: {en}")
    if not keywords_en:
        keywords_en = [query]

    # 联想词并入关键词池 (核心词在前 → [:1]源取主词, [:2]/[:3]源取主词+联想词)。
    # 不常见选题: 用户措辞≠领域标准术语, 联想词正是标准术语的补充覆盖。
    kw_en = (keywords_en + (related_en or [])[:2])[:3]
    kw_cn = ((keywords_cn or [query]) + (related_cn or [])[:2])[:3]

    # 每源独立超时（来自 config.py SEARCH_TIMEOUT_*）
    import config

    async def _safe(fn, *a, timeout_s=10, **kw):
        try:
            return await asyncio.wait_for(fn(*a, **kw), timeout=timeout_s)
        except (asyncio.TimeoutError, Exception):
            return []

    results = await asyncio.gather(
        _safe(_search_arxiv, kw_en, timeout_s=config.SEARCH_TIMEOUT_ARXIV),
        _safe(_search_openalex, kw_en, timeout_s=config.SEARCH_TIMEOUT_OPENALEX),
        _safe(_search_openalex_cn, kw_cn, timeout_s=config.SEARCH_TIMEOUT_OPENALEX_CN),
        _safe(_search_s2, kw_en, timeout_s=config.SEARCH_TIMEOUT_S2),
        _safe(_search_core, kw_en, timeout_s=config.SEARCH_TIMEOUT_CORE),
        _safe(_search_crossref, kw_en, timeout_s=config.SEARCH_TIMEOUT_CROSSREF),
        _safe(_search_doaj, kw_en, timeout_s=config.SEARCH_TIMEOUT_DOAJ),
        _safe(_search_epmc, kw_en, timeout_s=config.SEARCH_TIMEOUT_EPMC),
        _safe(
            _search_pubmed, kw_en, timeout_s=config.SEARCH_TIMEOUT_PUBMED
        ),  # PubMed — 36M biomedical+CS
        _safe(_search_pwc, kw_en, timeout_s=config.SEARCH_TIMEOUT_PWC),
        _safe(
            _search_google_scholar_apify,
            kw_en,
            timeout_s=config.SEARCH_TIMEOUT_GS_APIFY,
        ),
        _safe(_make_cnki_async, query, kw_cn, timeout_s=config.SEARCH_TIMEOUT_CNKI),
    )
    arxiv_papers = results[0] if not isinstance(results[0], BaseException) else []
    oa_papers = results[1] if not isinstance(results[1], BaseException) else []
    cn_papers = results[2] if not isinstance(results[2], BaseException) else []
    s2_papers = results[3] if not isinstance(results[3], BaseException) else []
    core_papers = results[4] if not isinstance(results[4], BaseException) else []
    crossref_papers = results[5] if not isinstance(results[5], BaseException) else []
    doaj_papers = results[6] if not isinstance(results[6], BaseException) else []
    epmc_papers = results[7] if not isinstance(results[7], BaseException) else []
    pubmed_papers = results[8] if not isinstance(results[8], BaseException) else []
    pwc_papers = results[9] if not isinstance(results[9], BaseException) else []
    gs_papers = results[10] if not isinstance(results[10], BaseException) else []
    cnki_papers = results[11] if not isinstance(results[11], BaseException) else []

    # 合并去重 + 屏蔽过滤 + 语言检测 + CN 相关性过滤
    cn_kw_set = set((keywords_cn or [query]) + [query])

    def _cn_relevant(title: str, strict: bool = True) -> bool:
        """strict=True要求完整关键词; strict=False含任一关键词字符即通过"""
        if strict:
            for kw in cn_kw_set:
                if len(kw) >= 2 and kw in title:
                    return True
            return False
        cn_chars = {c for kw in cn_kw_set for c in kw if 19968 <= ord(c) <= 40869}
        return any(c in cn_chars for c in (title or ""))

    seen: set[str] = set()
    papers: list[PaperMeta] = []

    def _dedup_key(p: PaperMeta) -> str:
        """去重键: DOI 优先 (唯一标识), 否则标题规范化前 60 字符 (DeepSeek 审查 #6)。"""
        if p.doi:
            return f"doi:{p.doi.lower().strip()}"
        return (p.title or "").lower().strip()[:60]

    all_papers = (
        cnki_papers
        + gs_papers
        + cn_papers
        + oa_papers
        + s2_papers
        + arxiv_papers
        + core_papers
        + crossref_papers
        + doaj_papers
        + epmc_papers
        + pubmed_papers
        + pwc_papers
    )
    for p in all_papers:
        if _should_drop(p):
            continue
        # 中文论文必须与搜索关键词相关（前30篇严格，之后宽松）
        cn_count_done = sum(1 for pp in papers if _detect_cn(pp.title))
        if _detect_cn(p.title) and not _cn_relevant(
            p.title, strict=(cn_count_done < 30)
        ):
            continue
        key = _dedup_key(p)
        if key and key not in seen:
            seen.add(key)
            papers.append(p)

    # CN 宽松补充：如果严格过滤后 < 20 篇中文，用宽松模式补
    cn_strict = sum(1 for p in papers if _detect_cn(p.title))
    if cn_strict < 20:
        for p in cn_papers:
            if (
                p not in papers
                and not _should_drop(p)
                and _detect_cn(p.title)
                and _cn_relevant(p.title, strict=False)
            ):
                key = _dedup_key(p)
                if key not in seen:
                    seen.add(key)
                    papers.append(p)

    def _sort_papers():
        """质量排序：中文高质量 > 英文高质量 > 低质量。
        用真实来源权重 (修复硬编码 "other" 导致 SOURCE_WEIGHTS 失效)。"""
        for p in papers:
            p.is_cn = _detect_cn(p.title)
            p._score = _quality_score(p)
        papers.sort(
            key=lambda p: (
                not bool(p.is_cn),  # 中文优先
                not p.is_oa,  # OA 优先
                -(p._score or 0),  # 质量分降序 (来源权重+引用+年份)
                -(p.year or 0),  # 年份降序
            )
        )

    # 首次排序
    _sort_papers()

    # Unpaywall 补充 PDF 直链
    enriched = await _enrich_via_unpaywall(papers)
    # 摘要补全：对无摘要论文，通过 Crossref/OpenAlex 补全
    abs_enriched = await _enrich_abstracts(papers)

    # 补全后重排 — is_oa/摘要变化会影响排序 (修复补全在排序后失效的缺陷)
    _sort_papers()

    logger.info(
        f"Search: {len(papers)} papers, {enriched} PDFs via Unpaywall, {abs_enriched} abstracts filled"
    )

    # 主题相关度重排 (修复: 原仅垃圾词过滤 + 质量分排序, 无关论文混入前列;
    # 按 query 与 标题+摘要 的嵌入余弦相似度降序, 只重排不删除)
    # 关键: 重排查询用「原文 + 英文关键词」双语表示 — 否则中文查询对英文论文的
    # 相似度偏低, 通用中文神经网络论文会压过真正相关的 DARTS 等英文论文。
    rerank_query = ((query or "") + " " + " ".join(keywords_en or [])).strip()
    papers = _rerank_by_relevance(papers, rerank_query)
    _cache_set(_SEARCH_CACHE, _cache_key, [p.model_dump() for p in papers])
    _save_caches()
    return papers


def _rerank_by_relevance(papers: list, query: str, top_k: int = 0) -> list:
    """按 query 与 标题+摘要 的语义相似度降序重排搜索结果。
    只重排不删除 (top_k=0 保留全部); 嵌入不可用时返回原序 (鲁棒降级)。"""
    if not papers:
        return papers
    try:
        import numpy as np

        from embedder import embed_texts

        texts = [(p.title or "")[:150] + " " + (p.abstract or "")[:600] for p in papers]
        vecs = np.asarray(embed_texts([query] + texts))
        sims = vecs[1:] @ vecs[0]
        # 语义相似度归一化到 [0,1]
        s_min, s_max = float(sims.min()), float(sims.max())
        sim_norm = (sims - s_min) / (s_max - s_min + 1e-9)
        # 引用信号: 对数压缩 (1引≈0, 100引≈0.46, 1万引≈0.85), 避免被高被引无关论文带偏
        cits = np.array([float(p.citation_count or 0) for p in papers])
        cit_log = np.log1p(cits) / np.log1p(10000.0)
        # 语义为主(0.9) + 引用小幅提权(0.1): 只在同样相关的论文里让高被引的靠前,
        # 绝不让 MapReduce(1.8万引) 这类无关论文压过真正相关的 DARTS 论文
        scores = 0.9 * sim_norm + 0.1 * cit_log
        # 知网占位条目沉底: 非真实论文, 只是 CNKI 跳转链接 (保留入口但不污染前列)
        for i, p in enumerate(papers):
            if (p.title or "").startswith("[知网]") or (p.title or "").startswith(
                "[CNKI]"
            ):
                scores[i] = -1.0
        ranked = sorted(zip(scores, papers), key=lambda x: -x[0])
        out = [p for _, p in ranked]
        return out[:top_k] if top_k > 0 else out
    except Exception:
        return papers


async def _search_s2(keywords: list[str]) -> list[PaperMeta]:
    """Semantic Scholar — always try, rate limit is 1/s without key. 题目+摘要质量最高。"""
    papers: list[PaperMeta] = []
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    async with httpx.AsyncClient(timeout=12.0) as client:
        for kw in keywords[:1]:  # only 1 kw, S2 is rich enough
            try:
                resp = await client.get(
                    S2_URL,
                    params={"query": kw, "limit": 20, "fields": S2_FIELDS},
                    headers=headers,
                )
                if resp.status_code == 429:
                    if not SEMANTIC_SCHOLAR_API_KEY:
                        logger.debug("S2 rate limited (no key)")
                        return papers
                    await asyncio.sleep(2)
                    continue
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("data", []):
                    pid = item.get("paperId", "")
                    if not pid:
                        continue
                    ext = item.get("externalIds") or {}
                    oa = item.get("openAccessPdf") or {}
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=item.get("title", "Unknown"),
                            authors=", ".join(
                                a.get("name", "") for a in (item.get("authors") or [])
                            ),
                            year=item.get("year"),
                            abstract=(item.get("abstract") or "")[:300],
                            doi=ext.get("DOI"),
                            arxiv_id=ext.get("ArXiv"),
                            pdf_url=oa.get("url"),
                            is_oa=bool(oa.get("url")),
                            citation_count=item.get("citationCount") or 0,
                        )
                    )
                break
            except Exception:
                logger.debug(f"S2 failed for '{kw}'")
    return papers


async def _search_arxiv(keywords: list[str]) -> list[PaperMeta]:
    """arXiv API 搜索。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for kw in keywords[:2]:  # 只用第一个关键词，arXiv API 慢
            # arXiv 简单搜索：直接用关键词
            terms = kw  # arXiv accepts plain space-separated keywords
            try:
                resp = await client.get(
                    ARXIV_URL,
                    params={
                        "search_query": terms,
                        "max_results": 30,
                        "sortBy": "relevance",
                    },
                )
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    aid = (
                        (entry.find("atom:id", ns).text or "")
                        .split("/abs/")[-1]
                        .split("v")[0]
                    )
                    if not aid:
                        continue
                    title_el = entry.find("atom:title", ns)
                    summary_el = entry.find("atom:summary", ns)
                    pub_el = entry.find("atom:published", ns)
                    title = (
                        title_el.text.strip()
                        if title_el is not None and title_el.text
                        else "Unknown"
                    )
                    yr = (
                        int(pub_el.text[:4])
                        if pub_el is not None and pub_el.text
                        else None
                    )
                    author_names = [
                        a.find("atom:name", ns).text
                        for a in entry.findall("atom:author", ns)
                        if a.find("atom:name", ns) is not None
                    ]
                    papers.append(
                        PaperMeta(
                            paper_id=hashlib.md5(f"arxiv:{aid}".encode()).hexdigest()[
                                :16
                            ],
                            title=title,
                            authors=", ".join(n or "" for n in author_names),
                            year=yr,
                            abstract=(summary_el.text or "").strip()[:200]
                            if summary_el is not None and summary_el.text
                            else "",
                            arxiv_id=aid,
                            pdf_url=f"https://arxiv.org/pdf/{aid}.pdf",
                            is_oa=True,
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
            await asyncio.sleep(0.05)
    return papers


# OpenAlex — 免费无限制，2.5 亿篇论文，返回 best_oa_location
OPENALEX_URL = "https://api.openalex.org/works"


async def _search_openalex(keywords: list[str]) -> list[PaperMeta]:
    """OpenAlex 搜索：免费、无速率限制、覆盖全面。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for kw in keywords[:3]:  # OpenAlex 快且准, 接住核心词+联想词
            try:
                resp = await client.get(
                    OPENALEX_URL,
                    params={
                        "search": kw,
                        "per_page": 60,
                        "sort": "cited_by_count:desc",
                        "filter": "has_doi:true",
                    },
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for item in data.get("results", []):
                    pid = (
                        item.get("id", "").split("/")[-1]
                        or hashlib.md5(str(item).encode()).hexdigest()[:16]
                    )
                    oa_loc = item.get("best_oa_location") or {}
                    oa_url = oa_loc.get("pdf_url") or oa_loc.get("landing_page_url")
                    authorships = item.get("authorships") or []
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=item.get("title", "Unknown"),
                            authors=", ".join(
                                (a.get("author", {}) or {}).get("display_name", "")
                                for a in authorships
                            ),
                            year=item.get("publication_year"),
                            abstract="",  # OpenAlex 不直接返回摘要
                            doi=item.get("doi", "").replace("https://doi.org/", ""),
                            pdf_url=oa_url,
                            is_oa=bool(oa_url)
                            or (item.get("open_access", {}) or {}).get("is_oa", False),
                            citation_count=item.get("cited_by_count") or 0,
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
            await asyncio.sleep(0.1)
    return papers


async def _search_openalex_cn(keywords: list[str]) -> list[PaperMeta]:
    """OpenAlex 中文论文搜索：filter=language:zh。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for kw in keywords[:3]:  # 搜所有中文关键词
            try:
                resp = await client.get(
                    OPENALEX_URL,
                    params={
                        "search": kw,
                        "per_page": 60,
                        "sort": "cited_by_count:desc",
                        "filter": "language:zh,has_doi:true",
                    },
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for item in data.get("results", []):
                    pid = (
                        item.get("id", "").split("/")[-1]
                        or hashlib.md5(str(item).encode()).hexdigest()[:16]
                    )
                    oa_loc = item.get("best_oa_location") or {}
                    oa_url = oa_loc.get("pdf_url") or oa_loc.get("landing_page_url")
                    authorships = item.get("authorships") or []
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=item.get("title", "Unknown"),
                            authors=", ".join(
                                (a.get("author", {}) or {}).get("display_name", "")
                                for a in authorships
                            ),
                            year=item.get("publication_year"),
                            abstract="",
                            doi=item.get("doi", "").replace("https://doi.org/", ""),
                            pdf_url=oa_url,
                            is_oa=bool(oa_url)
                            or (item.get("open_access", {}) or {}).get("is_oa", False),
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
            await asyncio.sleep(0.1)
    return papers


# ---- Apify 谷歌学术搜索 ----
APIFY_GS_URL = "https://api.apify.com/v2/acts/apify~google-scholar-scraper/runs"


async def _search_google_scholar_apify(keywords: list[str]) -> list[PaperMeta]:
    """通过 Apify 调用谷歌学术搜索（需要 APIFY_API_KEY）。

    v5.3 修复 (DeepSeek 审查 #1): 原实现 POST 后立即解析结果——但 Apify /runs 端点
    返回的是运行对象而非搜索结果, 导致永远拿不到论文。现改为正确流程:
    POST 启动运行 → 轮询运行状态 → 拉取数据集结果。"""
    if not APIFY_API_KEY:
        return []
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for kw in keywords[:2]:
            try:
                # Step 1: 启动 Apify actor 运行 (不等待, 获取 run_id)
                resp = await client.post(
                    APIFY_GS_URL,
                    json={"keyword": kw, "maxPages": 1, "maxResults": 20},
                    headers={"Authorization": f"Bearer {APIFY_API_KEY}"},
                )
                if resp.status_code != 201:
                    logger.warning(f"Apify GS start failed: HTTP {resp.status_code}")
                    continue
                run_data = resp.json().get("data", {})
                run_id = run_data.get("id")
                if not run_id:
                    logger.warning("Apify GS: no run_id in response")
                    continue

                # Step 2: 轮询运行状态 (最多 ~25s, 避免阻塞整个并行组)
                status = None
                for _ in range(10):
                    await asyncio.sleep(2.5)
                    status_resp = await client.get(
                        f"https://api.apify.com/v2/actor-runs/{run_id}",
                        headers={"Authorization": f"Bearer {APIFY_API_KEY}"},
                    )
                    if status_resp.status_code == 200:
                        status = status_resp.json().get("data", {}).get("status")
                        if status in ("SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"):
                            break
                if status != "SUCCEEDED":
                    logger.warning(f"Apify GS run {run_id} ended with status: {status}")
                    continue

                # Step 3: 拉取数据集结果
                dataset_id = run_data.get("defaultDatasetId")
                if dataset_id:
                    items_resp = await client.get(
                        f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                        headers={"Authorization": f"Bearer {APIFY_API_KEY}"},
                    )
                    if items_resp.status_code == 200:
                        for item in items_resp.json()[:20]:
                            title = item.get("title", "")
                            if not title:
                                continue
                            pid = hashlib.md5(f"gs_apify:{title}".encode()).hexdigest()[
                                :16
                            ]
                            papers.append(
                                PaperMeta(
                                    paper_id=pid,
                                    title=title,
                                    authors=item.get("authors", ""),
                                    year=int(item.get("year", 0))
                                    if item.get("year")
                                    else None,
                                    abstract=(item.get("description") or "")[:200],
                                    pdf_url=item.get("pdfUrl"),
                                    is_oa=bool(item.get("pdfUrl")),
                                )
                            )
            except Exception as e:
                logger.warning(f"Apify GS search failed: {e}")
    logger.info(f"Google Scholar (Apify): {len(papers)} papers")
    return papers


async def _make_cnki_async(query: str, keywords_cn: list[str]) -> list[PaperMeta]:
    """生成知网搜索占位条目。"""
    kw = "+".join(keywords_cn[:3]) if keywords_cn else query
    cnki_url = f"https://kns.cnki.net/kns8/defaultresult/index?kwd={kw}"
    return [
        PaperMeta(
            paper_id=hashlib.md5(f"cnki:{query}".encode()).hexdigest()[:16],
            title=f"[知网] 搜索「{' '.join(keywords_cn[:3]) if keywords_cn else query}」",
            abstract=f"点击跳转知网检索。关键词：{'、'.join(keywords_cn[:3]) if keywords_cn else query}。登录学校账号后可下载全文。",
            is_oa=False,
            pdf_url=cnki_url,
        )
    ]


# ---------- Unpaywall PDF 直链补充 ----------

UNPAYWALL_EMAIL = "research@academic-assistant.io"


async def _enrich_abstracts(papers: list[PaperMeta]) -> int:
    """对无摘要论文, 并行通过 Crossref 补全摘要 (限 10 篇, 3s 超时)."""
    candidates = [
        p for p in papers if (not p.abstract or len(p.abstract) < 30) and p.doi
    ][:10]
    if not candidates:
        return 0

    filled = 0
    async with httpx.AsyncClient(timeout=5.0) as client:

        async def _one(p):
            try:
                resp = await client.get(
                    f"https://api.crossref.org/works/{p.doi}",
                    headers={
                        "User-Agent": "AI-Research-Assistant/1.0 (mailto:research@example.com)"
                    },
                )
                if resp.status_code == 200:
                    msg = resp.json().get("message", {})
                    abstract = msg.get("abstract", "")
                    if abstract and len(abstract) > 30:
                        # Strip HTML tags from Crossref abstract
                        import re

                        abstract = re.sub(r"<[^>]+>", "", abstract)[:500]
                        p.abstract = abstract
                        return 1
            except Exception:
                logger.debug("source search failed", exc_info=True)
            return 0

        results = await asyncio.gather(
            *[_one(p) for p in candidates], return_exceptions=True
        )
        filled = sum(r for r in results if isinstance(r, int))
    return filled


async def _enrich_via_unpaywall(papers: list[PaperMeta]) -> int:
    """对无 pdf_url 但有 DOI 的论文，并行调 Unpaywall 查合法免费 PDF（限5篇,3s超时）。
    免费 API，每分钟 100 次，无需注册。"""
    candidates = [p for p in papers if not p.pdf_url and p.doi][:15]
    if not candidates:
        return 0

    async with httpx.AsyncClient(timeout=5.0) as client:

        async def _one(p):
            try:
                resp = await client.get(
                    f"https://api.unpaywall.org/v2/{p.doi}",
                    params={"email": UNPAYWALL_EMAIL},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    best = data.get("best_oa_location") or {}
                    u = best.get("url_for_pdf") or best.get("url")
                    if u:
                        p.pdf_url = u
                        p.is_oa = True
                        return 1
            except Exception:
                logger.debug("source search failed", exc_info=True)
            return 0

        results = await asyncio.gather(
            *[_one(p) for p in candidates], return_exceptions=True
        )
        enriched = sum(r for r in results if isinstance(r, int))
    return enriched


# ---- CORE API (free, 30M+ OA papers) ----
CORE_URL = "https://api.core.ac.uk/v3/search/works"


async def _search_core(keywords: list[str]) -> list[PaperMeta]:
    """CORE.ac.uk — 30M+ OA论文聚合器。免费API, 30s超时。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=12.0) as client:
        for kw in keywords[:1]:
            try:
                resp = await client.get(
                    CORE_URL,
                    params={"q": kw, "limit": 15, "scroll": "false"},
                    headers={"User-Agent": "AI-Research-Assistant/1.0"},
                )
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("results", []):
                    pid = f"core:{item.get('id','')}"
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=item.get("title", "Unknown"),
                            authors=", ".join(
                                a.get("name", "") for a in (item.get("authors") or [])
                            ),
                            year=item.get("yearPublished"),
                            abstract=(item.get("abstract") or "")[:300],
                            doi=item.get("doi"),
                            is_oa=True,
                            pdf_url=item.get("downloadUrl")
                            or item.get("sourceFulltextUrls", [None])[0],
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
    return papers


# ---- Crossref API (free, 140M+ DOI metadata) ----
CROSSREF_URL = "https://api.crossref.org/works"


async def _search_crossref(keywords: list[str]) -> list[PaperMeta]:
    """Crossref — 140M+ DOI元数据。免费API, 礼貌使用。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for kw in keywords[:1]:
            try:
                resp = await client.get(
                    CROSSREF_URL,
                    params={"query": kw, "rows": 15, "filter": "type:journal-article"},
                    headers={
                        "User-Agent": "AI-Research-Assistant/1.0 (mailto:research@example.com)"
                    },
                )
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("message", {}).get("items", []):
                    pid = f"crossref:{item.get('DOI','')}"
                    authors = item.get("author", [])
                    yr = (
                        item.get("published-print", {}).get("date-parts", [[None]])[0][
                            0
                        ]
                        or item.get("created", {}).get("date-parts", [[None]])[0][0]
                    )
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=(item.get("title") or ["Unknown"])[0],
                            authors=", ".join(
                                (a.get("family", "") or "") for a in authors
                            ),
                            year=yr,
                            abstract=item.get("abstract", "")[:300],
                            doi=item.get("DOI"),
                            is_oa=(item.get("license") or [{}])[0]
                            .get("URL", "")
                            .find("creativecommons")
                            >= 0,
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
    return papers


# ---- DOAJ API (free, all OA journals) ----
DOAJ_URL = "https://doaj.org/api/search/articles"


async def _search_doaj(keywords: list[str]) -> list[PaperMeta]:
    """DOAJ — Directory of Open Access Journals。全部免费。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for kw in keywords[:1]:
            try:
                resp = await client.get(
                    DOAJ_URL,
                    params={"q": kw, "pageSize": 15},
                )
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("results", []):
                    bib = item.get("bibjson", {})
                    pid = f"doaj:{item.get('id','')}"
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=bib.get("title", "Unknown"),
                            authors=", ".join(
                                a.get("name", "") for a in bib.get("author", [])
                            ),
                            year=int(bib.get("year", 0)) if bib.get("year") else None,
                            abstract=bib.get("abstract", "")[:300],
                            is_oa=True,
                            pdf_url=next(
                                (
                                    lnk["url"]
                                    for lnk in bib.get("link", [])
                                    if lnk.get("type") == "fulltext"
                                ),
                                None,
                            ),
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
    return papers


# ---- PubMed (NCBI E-utilities, free, no key, 36M+ biomedical + CS) ----
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


async def _search_pubmed(keywords: list[str]) -> list[PaperMeta]:
    """PubMed — free API, 36M+ biomedical + interdisciplinary CS papers."""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for kw in keywords[:1]:
            try:
                # Step 1: search IDs
                resp = await client.get(
                    PUBMED_SEARCH_URL,
                    params={
                        "db": "pubmed",
                        "term": kw,
                        "retmax": 15,
                        "retmode": "json",
                    },
                )
                if resp.status_code != 200:
                    continue
                ids = resp.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    continue

                # Step 2: fetch summaries
                resp2 = await client.get(
                    PUBMED_SUMMARY_URL,
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                )
                if resp2.status_code != 200:
                    continue
                results = resp2.json().get("result", {})
                for uid in ids:
                    item = results.get(uid, {})
                    if not item:
                        continue
                    doi = None
                    for aid in item.get("articleids", []):
                        if aid.get("idtype") == "doi":
                            doi = aid.get("value")
                    papers.append(
                        PaperMeta(
                            paper_id=f"pubmed:{uid}",
                            title=item.get("title", "Unknown"),
                            authors=", ".join(
                                a.get("name", "") for a in item.get("authors", [])
                            ),
                            year=int(str(item.get("pubdate", ""))[:4])
                            if item.get("pubdate")
                            else None,
                            abstract="",  # PubMed summary doesn't include full abstract
                            doi=doi,
                            is_oa=False,
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
            await asyncio.sleep(0.35)  # NCBI rate limit: 3/sec without key
    return papers


# ---- Papers With Code (free API, CS papers with code links) ----
PWC_URL = "https://paperswithcode.com/api/v1/papers/"


async def _search_pwc(keywords: list[str]) -> list[PaperMeta]:
    """Papers With Code — free API, CS papers with implementation links."""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for kw in keywords[:1]:
            try:
                resp = await client.get(
                    PWC_URL,
                    params={"q": kw, "items_per_page": 15},
                )
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("results", []):
                    pid = f"pwc:{item.get('id','')}"
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=item.get("title", "Unknown"),
                            authors="",  # PWC basic API doesn't return authors
                            year=item.get("published"),  # YYYY-MM-DD format
                            abstract=item.get("abstract", "")[:300],
                            doi=None,
                            is_oa=False,
                            # Note: paper_url is the PWC page with code links
                            pdf_url=item.get("url_abs"),
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
    return papers


# ---- Europe PMC (free, 41M+ life sciences, richer metadata than PubMed) ----
EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


async def _search_epmc(keywords: list[str]) -> list[PaperMeta]:
    """Europe PMC — 41M+ life science papers, free API. OA标志+PDF链接。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for kw in keywords[:1]:
            try:
                resp = await client.get(
                    EPMC_URL,
                    params={
                        "query": kw,
                        "resultType": "core",
                        "pageSize": 20,
                        "format": "json",
                    },
                )
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("resultList", {}).get("result", []):
                    doi = item.get("doi")
                    pdfl = item.get("hasPDF") if item.get("hasPDF") == "Y" else None
                    papers.append(
                        PaperMeta(
                            paper_id=f"epmc:{item.get('id','')}",
                            title=item.get("title", "Unknown"),
                            authors=item.get("authorString", ""),
                            year=int(item.get("pubYear", 0))
                            if item.get("pubYear")
                            else None,
                            abstract=(item.get("abstractText") or "")[:300],
                            doi=doi,
                            is_oa=item.get("isOpenAccess") == "Y",
                            pdf_url=f"https://europepmc.org/articles/{item.get('pmcid')}/pdf"
                            if item.get("pmcid") and pdfl
                            else None,
                        )
                    )
            except Exception:
                logger.debug("source search failed", exc_info=True)
    return papers
