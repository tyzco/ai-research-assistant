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
    SEMANTIC_SCHOLAR_API_KEY,
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


async def generate_search_strategy(query: str) -> dict:
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": STRATEGY_PROMPT.format(query=query)}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    import re

    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed: {text[:200]}")
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


# ---------- 质量过滤 & 语言检测 ----------


def _detect_cn(text: str) -> bool:
    """charCode 汉字检测（与前端 isCN 一致）。"""
    for c in text or "":
        if 19968 <= ord(c) <= 40869:
            return True
    return False


# 来源权重：越高越可信
_SRC_W = {
    "Semantic Scholar": 10,
    "OpenAlex": 8,
    "arXiv": 7,
    "baidu_xueshu": 3,
    "nssd": 5,
}

# 屏蔽关键词（含这些的论文直接丢弃）
_BLOCK = [
    "征稿",
    "会议通知",
    "征文",
    "订阅",
    "广告",
    "约稿",
    "稿约",
    "投稿须知",
    "撤稿",
    "抄袭",
    "学术不端",
    "书评",
    "读者来信",
    "简讯",
    "新闻",
]


def _quality_score(p: PaperMeta, source: str = "") -> float:
    """综合质量评分：来源权重 + 引用数 + 年份。"""
    w = _SRC_W.get(source, 2) if source else 2
    cit = p.year or 0
    return w * 10 + max(0, (p.year or 2000) - 2000) * 0.1


def _should_drop(p: PaperMeta) -> bool:
    """仅丢弃明显垃圾（广告/撤稿/征稿）。保留其他所有论文。"""
    t = (p.title or "") + (p.abstract or "")
    spam_kw = ["征稿", "广告", "撤稿", "抄袭", "学术不端", "约稿", "稿约"]
    for w in spam_kw:
        if w in t:
            return True
    # 标题短于4字且无DOI才丢弃
    if len(p.title or "") < 4 and not p.doi:
        return True
    return False


# ---------- 论文搜索（并行 S2 + arXiv + CNKI 链接）----------

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "title,authors,year,abstract,openAccessPdf,externalIds,journal,citationCount"
)
ARXIV_URL = "https://export.arxiv.org/api/query"


async def search_papers_for_topic(
    query: str,
    keywords_en: list[str] | None = None,
    keywords_cn: list[str] | None = None,
) -> list[PaperMeta]:
    """9 源并行搜索 + Query扩展: 每源独立超时, 快源先回。参考 academic-search 的 Query 扩展策略。"""
    if not keywords_en:
        strategy = await generate_search_strategy(query)
        keywords_en = strategy.get("keywords_en", [])[:2]
        keywords_cn = strategy.get("keywords_cn", [])[:3]
    if not keywords_en:
        keywords_en = [query]

    # Query扩展: 每个关键词搜索2个变体(同义词/缩写), 覆盖提升30-50%。来源: academic-search v1.2.0
    kw_en = keywords_en[:1]
    kw_cn = (keywords_cn or [query])[:1]
    # 用 related_terms 扩展搜索面(如果从策略生成了)
    kw_en_expanded = (
        list(
            set(
                kw_en
                + (
                    strategy.get("related_terms_en", [])[:2]
                    if "strategy" in dir()
                    else []
                )
            )
        )
        if keywords_en
        else kw_en
    )
    kw_cn_expanded = (
        list(
            set(
                kw_cn
                + (
                    strategy.get("related_terms_cn", [])[:2]
                    if "strategy" in dir()
                    else []
                )
            )
        )
        if keywords_cn
        else kw_cn
    )

    # 每源独立超时: arXiv/OpenAlex=8s, S2/Apify=3s, CNKI=2s
    async def _safe(fn, *a, timeout_s=10, **kw):
        try:
            return await asyncio.wait_for(fn(*a, **kw), timeout=timeout_s)
        except:
            return []

    results = await asyncio.gather(
        _safe(_search_arxiv, kw_en, timeout_s=10),
        _safe(_search_openalex, kw_en, timeout_s=10),
        _safe(_search_openalex_cn, kw_cn, timeout_s=10),
        _safe(_search_s2, kw_en, timeout_s=5),
        _safe(_search_core, kw_en, timeout_s=8),
        _safe(_search_crossref, kw_en, timeout_s=8),
        _safe(_search_doaj, kw_en, timeout_s=8),
        _safe(_search_epmc, kw_en, timeout_s=8),
        _safe(_search_epmc, kw_en, timeout_s=8),
        _safe(_search_pubmed, kw_en, timeout_s=8),  # PubMed — 36M biomedical+CS
        _safe(_search_pwc, kw_en, timeout_s=8),  # PapersWithCode — CS+code
        _safe(_search_google_scholar_apify, kw_en, timeout_s=3),
        _safe(_make_cnki_async, query, kw_cn, timeout_s=2),
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
    all_papers = (
        cnki_papers
        + gs_papers
        + cn_papers
        + oa_papers
        + s2_papers
        + arxiv_papers
        + core_papers
        + crossref_papers
        + doaj_papers + epmc_papers + pubmed_papers + pwc_papers
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
        key = (p.title or "").lower().strip()[:60]
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
                key = (p.title or "").lower().strip()[:60]
                if key not in seen:
                    seen.add(key)
                    papers.append(p)

    # 质量排序：中文高质量 > 英文高质量 > 低质量
    for p in papers:
        p.is_cn = _detect_cn(p.title)
        p._score = _quality_score(p, "other")

    papers.sort(
        key=lambda p: (
            not bool(p.is_cn),  # 中文优先
            not p.is_oa,  # OA 优先
            -(p._score or 0),  # 质量分降序
            -(p.year or 0),  # 年份降序
        )
    )
    # Unpaywall 补充 PDF 直链（对无 pdf_url 但有 DOI 的论文）
    enriched = await _enrich_via_unpaywall(papers)
    logger.info(
        f"Search: {len(papers)} papers (CNKI:{len(cnki_papers)} GS:{len(gs_papers)} CN:{len(cn_papers)} OA:{len(oa_papers)} S2:{len(s2_papers)} arXiv:{len(arxiv_papers)}), "
        f"{enriched} enriched via Unpaywall"
    )
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
                pass
            await asyncio.sleep(0.05)
    return papers


# OpenAlex — 免费无限制，2.5 亿篇论文，返回 best_oa_location
OPENALEX_URL = "https://api.openalex.org/works"


async def _search_openalex(keywords: list[str]) -> list[PaperMeta]:
    """OpenAlex 搜索：免费、无速率限制、覆盖全面。"""
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for kw in keywords[:2]:
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
                        )
                    )
            except Exception:
                pass
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
                pass
            await asyncio.sleep(0.1)
    return papers


# ---- Apify 谷歌学术搜索 ----
APIFY_GS_URL = "https://api.apify.com/v2/acts/apify~google-scholar-scraper/runs"


async def _search_google_scholar_apify(keywords: list[str]) -> list[PaperMeta]:
    """通过 Apify 调用谷歌学术搜索（需要 APIFY_API_KEY）。"""
    if not APIFY_API_KEY:
        return []
    papers: list[PaperMeta] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for kw in keywords[:2]:
            try:
                resp = await client.post(
                    APIFY_GS_URL,
                    json={"keyword": kw, "maxPages": 1, "maxResults": 20},
                    headers={"Authorization": f"Bearer {APIFY_API_KEY}"},
                    params={"waitForFinish": 120},
                    timeout=120.0,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for item in data.get("data", [])[:20]:
                    title = item.get("title", "")
                    if not title:
                        continue
                    pid = hashlib.md5(f"gs_apify:{title}".encode()).hexdigest()[:16]
                    papers.append(
                        PaperMeta(
                            paper_id=pid,
                            title=title,
                            authors=item.get("authors", ""),
                            year=int(item.get("year", 0)) if item.get("year") else None,
                            abstract=(item.get("description") or "")[:200],
                            pdf_url=item.get("pdfUrl"),
                            is_oa=bool(item.get("pdfUrl")),
                        )
                    )
            except Exception:
                pass
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
                pass
            return 0

        results = await asyncio.gather(
            *[_one(p) for p in candidates], return_exceptions=True
        )
        enriched = sum(r for r in results if isinstance(r, int))
    return enriched
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
                pass
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
                pass
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
                pass
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
                pass
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
                pass
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
                    params={"query": kw, "resultType": "core", "pageSize": 20, "format": "json"},
                )
                if resp.status_code != 200: continue
                for item in resp.json().get("resultList", {}).get("result", []):
                    doi = item.get("doi")
                    pdfl = item.get("hasPDF") if item.get("hasPDF") == "Y" else None
                    papers.append(PaperMeta(
                        paper_id=f"epmc:{item.get('id','')}",
                        title=item.get("title", "Unknown"),
                        authors=item.get("authorString", ""),
                        year=int(item.get("pubYear", 0)) if item.get("pubYear") else None,
                        abstract=(item.get("abstractText") or "")[:300],
                        doi=doi, is_oa=item.get("isOpenAccess") == "Y",
                        pdf_url=f"https://europepmc.org/articles/{item.get('pmcid')}/pdf" if item.get("pmcid") and pdfl else None,
                    ))
            except Exception: pass
    return papers
