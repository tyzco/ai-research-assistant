"""FastAPI 独立服务（端口 8001）：检索策略 + 论文搜索 + PDF 上传 + 问答 + ZIP 批量下载。"""

import io
import uuid
import zipfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from auth import create_token, get_current_user, login_user, register_user, require_user
from config import CN_EN_TERMS, IMAGE_DIR, PROJECT_ROOT
from downloader import chunk_text, parse_pdf
from knowledge_base import build_knowledge_base, store_image_descriptions
from models import AskRequest, CreateTopicRequest, PaperMeta, TopicStatus
from monitor import metrics, sanitize_input
from search import search_papers_for_topic
from topic_manager import active_topics, create_topic

# 消息历史存储（按 topic_id）
message_store: dict[str, list[dict]] = {}

app = FastAPI(title="AI Research Assistant", version="0.3.0")


@app.on_event("startup")
async def startup_preload():
    """预热：提前加载嵌入模型（非阻塞，失败不影响启动）。"""
    import asyncio
    import logging

    logger = logging.getLogger("startup")

    def _do_preload():
        try:
            from knowledge_base import _get_embed_model

            _get_embed_model()
            logger.info("Embedding model preloaded")
        except Exception as e:
            logger.warning(f"Embed preload failed: {e}")
        try:
            from sentence_transformers import CrossEncoder

            CrossEncoder("BAAI/bge-reranker-base", max_length=512)
            logger.info("Reranker model preloaded")
        except Exception as e:
            logger.warning(f"Reranker preload failed: {e}")

    asyncio.get_event_loop().run_in_executor(None, _do_preload)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

import time as _time

# ---- 简易限流: 每 IP 最多 30 req/s ----
from collections import defaultdict

_rate_limiter: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = _time.time()
    window = [t for t in _rate_limiter[ip] if now - t < 1.0]
    if len(window) > 30:
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "请求过于频繁，请稍后重试"}, status_code=429)
    window.append(now)
    _rate_limiter[ip] = window[-100:]  # trim old entries
    return await call_next(request)


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def api_metrics():
    """系统监控指标。"""
    return metrics.to_dict()


@app.get("/kg_data")
async def api_kg_data(table: str = ""):
    """知识图谱数据端点。table 可以是 topic_id 或 LanceDB table 名。"""
    import lancedb

    from config import LANCEDB_DIR
    from knowledge_graph import build_graph_from_kb

    db = lancedb.connect(str(LANCEDB_DIR))
    raw = db.list_tables()
    all_tables = (
        list(raw.tables)
        if hasattr(raw, "tables")
        else (list(raw[0]) if isinstance(raw, tuple) else list(raw))
    )

    # If table is a topic_id (12-char hex), resolve to lancedb table
    lancedb_table = table
    if table and len(table) == 12:
        state = active_topics.get(table)
        if state and state.lancedb_table:
            lancedb_table = state.lancedb_table
        elif table in all_tables:
            lancedb_table = table  # happens to be a table name too
        else:
            # fallback: find largest table
            pass

    if lancedb_table and lancedb_table in all_tables:
        return build_graph_from_kb(str(lancedb_table))

    # Fallback: find largest KB
    best, best_n = "", 0
    for t in all_tables:
        try:
            n = db.open_table(t).count_rows()
            if n > best_n:
                best, best_n = t, n
        except:
            pass
    if not best:
        return {"nodes": [], "links": []}
    return build_graph_from_kb(str(best))


@app.post("/kg/build")
async def api_kg_build(request: Request):
    """LLM 实体抽取 + 图谱构建：从 KB 论文摘要中提取方法/数据集/指标和关系。
    耗时约 10-30s（取决于论文数量）。"""
    req = await request.json()
    table = req.get("table", "")
    if not table:
        # 找最大 KB
        import lancedb

        from config import LANCEDB_DIR

        db = lancedb.connect(str(LANCEDB_DIR))
        raw = db.list_tables()
        tables = raw.tables if hasattr(raw, "tables") else []
        best, best_n = "", 0
        for t in tables:
            try:
                n = db.open_table(t).count_rows()
                if n > best_n:
                    best, best_n = t, n
            except:
                pass
        table = str(best) if best else ""
    if not table:
        raise HTTPException(404, "无可用知识库")

    from knowledge_graph import build_graph_from_kb, extract_entities_from_abstracts

    # 获取论文元数据
    graph = build_graph_from_kb(table)
    if not graph["nodes"]:
        raise HTTPException(404, "知识库无数据")

    # LLM 实体抽取
    abstracts = [
        {"paper_id": n["id"], "title": n["label"], "abstract": ""}
        for n in graph["nodes"][:20]
    ]
    extracted = await extract_entities_from_abstracts(abstracts)

    # 合并：论文节点 + 实体节点 + 关系边
    entity_ids = set()
    entities = []
    for e in extracted.get("entities", []):
        if e["id"] not in entity_ids:
            entity_ids.add(e["id"])
            entities.append(e)

    links = []
    for r in extracted.get("relations", []):
        links.append(
            {
                "source": r["source"],
                "target": r["target"],
                "type": r.get("relation", "related"),
            }
        )

    # 论文-论文共享实体边
    paper_methods = {}  # paper_id -> set of entity_ids
    for r in extracted.get("relations", []):
        src, tgt = r["source"], r["target"]
        if src not in paper_methods:
            paper_methods[src] = set()
        paper_methods[src].add(tgt)

    paper_ids = list(paper_methods.keys())
    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            shared = paper_methods[paper_ids[i]] & paper_methods[paper_ids[j]]
            if len(shared) >= 2:
                links.append(
                    {
                        "source": paper_ids[i],
                        "target": paper_ids[j],
                        "type": "shares_entity",
                        "count": len(shared),
                    }
                )

    return {
        "nodes": graph["nodes"] + entities,
        "links": links,
        "stats": {
            "papers": len(graph["nodes"]),
            "entities": len(entities),
            "relations": len(links),
        },
    }


# ===== 认证端点 =====

# Login rate limiting: 5 attempts per IP per minute
_login_attempts: dict[str, list[float]] = {}


@app.post("/register")
async def api_register(request: Request):
    req = await request.json()
    username = sanitize_input(req.get("username", ""))
    password = req.get("password", "")
    if not username or len(username) < 2 or len(username) > 32:
        raise HTTPException(400, "用户名需 2-32 个字符")
    if not password or len(password) < 6:
        raise HTTPException(400, "密码至少 6 个字符")
    ok, msg = register_user(username, password, req.get("email", ""))
    if not ok:
        raise HTTPException(400, msg)
    token = create_token(username)
    return {"access_token": token, "token_type": "bearer", "user": username}


@app.post("/login")
async def api_login(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = _time.time()
    window = [t for t in _login_attempts.get(ip, []) if now - t < 60]
    if len(window) >= 5:
        raise HTTPException(429, "登录尝试过于频繁，请 1 分钟后重试")
    req = await request.json()
    username = sanitize_input(req.get("username", ""))
    password = req.get("password", "")
    token = login_user(username, password)
    if not token:
        _login_attempts[ip] = window + [now]
        raise HTTPException(401, "用户名或密码错误")
    return {"access_token": token, "token_type": "bearer", "user": username}


@app.get("/me")
async def api_me(user_id: str = Depends(require_user)):
    return {"user_id": user_id}


@app.get("/topics")
async def api_list_topics(user_id: str = Depends(get_current_user)):
    # 多租户：只返回该用户的课题（TODO: 在 topic_manager 中按 user_id 隔离）
    return [
        {
            "topic_id": s.topic_id,
            "query": s.query,
            "status": s.status.value,
            "papers": s.total_papers,
        }
        for s in active_topics.values()
    ]


@app.delete("/topic/{topic_id}")
async def api_delete_topic(topic_id: str):
    if topic_id not in active_topics:
        raise HTTPException(404, "Topic not found")
    del active_topics[topic_id]
    from topic_manager import _save_topics

    _save_topics()
    return {"ok": True}


@app.get("/export/{topic_id}")
async def api_export_topic(topic_id: str):
    from fastapi.responses import Response

    state = active_topics.get(topic_id)
    if not state:
        raise HTTPException(404, "Topic not found")
    lines = [f"# {state.query}", "", f"创建时间: {state.created_at}"]
    if state.search_strategy:
        s = state.search_strategy
        lines += [
            "",
            "## 检索策略",
            "",
            f"中文关键词: {', '.join(s.get('keywords_cn', []))}",
            f"英文关键词: {', '.join(s.get('keywords_en', []))}",
        ]
        if s.get("domain_tags"):
            lines.append(f"领域: {', '.join(s['domain_tags'])}")
    lines += ["", "## 对话记录", ""]
    msgs = message_store.get(topic_id, [])
    for m in msgs:
        role = "👤 用户" if m["role"] == "user" else "🤖 AI"
        lines.append(f"### {role}")
        lines.append(m["content"])
        lines.append("")
    content = "\n".join(lines)
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=topic_{topic_id}.md"},
    )


async def _translate_cn_query(query: str) -> str:
    import re

    if not any(19968 <= ord(c) <= 40869 for c in query):
        return ""
    try:
        from openai import AsyncOpenAI

        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": f"将以下中文学术术语翻译为英文检索关键词，只返回英文逗号分隔，不要解释：{query}",
                }
            ],
            timeout=6,
        )
        text = (resp.choices[0].message.content or "").strip()
        return re.sub(r"[^a-zA-Z0-9,\- ]", "", text)[:100]
    except Exception:
        import traceback

        traceback.print_exc()
        return ""


def _quick_keywords(query: str) -> dict:
    import re

    words = [w.strip() for w in re.split(r"[，,、\s]+", query) if w.strip()]
    cn = [w for w in words if any(19968 <= ord(c) <= 40869 for c in w)]
    en = [w.lower() for w in words if w.isascii() and len(w) >= 2]
    en_from_dict = [v for k, v in CN_EN_TERMS.items() if k in query]
    if en_from_dict and en_from_dict[0] not in en:
        en = [en_from_dict[0]] + en
    keywords_cn = cn or [query]
    keywords_en = en if en else [query]
    bq = []
    if cn:
        bq.append(
            {
                "database": "\u77e5\u7f51",
                "query": " OR ".join(f"\u4e3b\u9898='{k}'" for k in cn[:3]),
                "note": "\u4e2d\u6587\u6838\u5fc3\u5173\u952e\u8bcd",
            }
        )
    if keywords_en:
        bq.append(
            {
                "database": "arXiv",
                "query": " OR ".join(keywords_en[:3]),
                "note": "\u82f1\u6587\u6838\u5fc3\u5173\u952e\u8bcd",
            }
        )
    return {
        "keywords_cn": keywords_cn,
        "keywords_en": keywords_en,
        "domain_tags": [],
        "related_terms_cn": [],
        "related_terms_en": [],
        "boolean_queries": bq,
        "recommended_databases": ["\u77e5\u7f51", "arXiv"] if cn else ["arXiv"],
        "top_authors": [],
        "search_tips": "",
    }


@app.post("/create_topic")
async def api_create_topic(req: CreateTopicRequest):
    """创建课题：本地关键词 + LLM翻译(非阻塞增强)。"""
    query = sanitize_input(req.query)
    state = create_topic(query, topic_id=req.topic_id if req.topic_id else "")
    strategy = _quick_keywords(query)
    state.search_strategy = strategy
    message_store[state.topic_id] = [{"role": "system", "content": f"研究方向: {query}"}]

    # 后台LLM翻译增强英文关键词（非阻塞）
    async def _enhance():
        try:
            en_llm = await _translate_cn_query(query)
            if en_llm:
                terms = [t.strip() for t in en_llm.split(",") if t.strip()]
                if terms:
                    strategy["keywords_en"] = list(set(terms + strategy["keywords_en"]))
                    strategy["boolean_queries"] = [
                        bq
                        for bq in strategy.get("boolean_queries", [])
                        if bq.get("database") != "arXiv"
                    ] + [
                        {
                            "database": "arXiv",
                            "query": " OR ".join(strategy["keywords_en"][:4]),
                            "note": "LLM翻译增强",
                        }
                    ]
                    state.search_strategy = strategy
        except Exception:
            pass

    import asyncio

    asyncio.create_task(_enhance())
    return {"topic_id": state.topic_id, "strategy": strategy}


@app.post("/search_papers/stream")
async def api_search_papers_stream(request: Request):
    """SSE流式论文搜索: 每批5篇渐进加载, 用户即刻看到结果。"""
    import asyncio
    import json as _json
    import time

    from fastapi.responses import StreamingResponse

    req = await request.json()
    query = req.get("query", "")
    keywords_en = req.get("keywords_en")
    keywords_cn = req.get("keywords_cn")
    if not query:
        raise HTTPException(400, "需要 query 参数")

    def _fmt(p):
        preview = dl = None
        if p.doi:
            preview = f"https://api.semanticscholar.org/DOI:{p.doi}"
            dl = p.pdf_url
        elif p.arxiv_id:
            preview = f"https://arxiv.org/abs/{p.arxiv_id}"
            dl = f"https://arxiv.org/pdf/{p.arxiv_id}.pdf"
        return {
            "paper_id": p.paper_id,
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "abstract": (p.abstract or "")[:500],
            "doi": p.doi,
            "arxiv_id": p.arxiv_id,
            "is_oa": p.is_oa,
            "pdf_url": p.pdf_url,
            "preview_url": preview,
            "download_url": dl,
            "source": _paper_source(p),
            "doi_url": f"https://doi.org/{p.doi}" if p.doi else None,
            "cnki_url": _cnki_search_url(p.title) if p.title else None,
            "google_scholar_url": _gs_search_url(p.title) if p.title else None,
            "scihub_url": f"https://sci-hub.se/{p.doi}" if p.doi else None,
        }

    async def generate():
        s = (
            lambda e, d: f"data: {_json.dumps({'event': e, **d}, ensure_ascii=False)}\n\n"
        )
        t0 = time.time()

        # Step 1: 快速返回第一批 (arXiv前5篇, 3s超时)
        from search import (
            _search_arxiv,
            _search_openalex,
            _search_openalex_cn,
            _should_drop,
        )

        async def _safe(fn, *a, to=5, **kw):
            try:
                return await asyncio.wait_for(fn(*a, **kw), timeout=to)
            except:
                return []

        kw_en = keywords_en[:1] if keywords_en else [query]
        kw_cn = keywords_cn[:1] if keywords_cn else [query]

        # Batch 1: arXiv (fastest, ~6s)
        batch1 = await _safe(_search_arxiv, kw_en, to=10)
        batch1 = [p for p in batch1 if not _should_drop(p)]
        yield s(
            "batch",
            {
                "source": "arxiv",
                "count": len(batch1),
                "papers": [_fmt(p) for p in batch1[:8]],
            },
        )

        # Batch 2: OpenAlex EN + CN (并行, ~8s)
        batch2 = await asyncio.gather(
            _safe(_search_openalex, kw_en, to=8),
            _safe(_search_openalex_cn, kw_cn, to=8),
        )
        oa_papers = [p for p in (batch2[0] or []) if not _should_drop(p)]
        cn_papers = [p for p in (batch2[1] or []) if not _should_drop(p)]
        # Filter CN relevance
        cn_filtered = [
            p
            for p in cn_papers
            if any(kw in (p.title or "") for kw in kw_cn if len(kw) >= 2)
        ]
        yield s(
            "batch",
            {
                "source": "openalex",
                "count": len(oa_papers) + len(cn_filtered),
                "papers": [_fmt(p) for p in (oa_papers + cn_filtered)[:12]],
            },
        )

        # Batch 3: 慢源 (S2 + CNKI, 不阻塞)
        from search import _make_cnki_async, _search_s2

        batch3 = await asyncio.gather(
            _safe(_search_s2, kw_en, to=3),
            _safe(_make_cnki_async, query, kw_cn, to=2),
        )
        s2_papers = [p for p in (batch3[0] or []) if not _should_drop(p)]
        cnki_papers = batch3[1] or []
        yield s(
            "batch",
            {
                "source": "s2+cnki",
                "count": len(s2_papers) + len(cnki_papers),
                "papers": [_fmt(p) for p in s2_papers[:5] + cnki_papers],
            },
        )

        total = (
            len(batch1)
            + len(oa_papers)
            + len(cn_filtered)
            + len(s2_papers)
            + len(cnki_papers)
        )
        yield s("done", {"total": total, "elapsed": f"{time.time()-t0:.1f}s"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/search_papers")
async def api_search_papers(request: Request):
    """搜索论文（兼容旧接口）。推荐使用 /search_papers/stream 流式分批返回。"""
    req = await request.json()
    query = req.get("query", "")
    keywords_en = req.get("keywords_en")
    keywords_cn = req.get("keywords_cn")
    papers = await search_papers_for_topic(query, keywords_en, keywords_cn)

    def _paper_preview(p):
        """Generate preview + download links for every paper."""
        preview = None
        dl = None
        if p.doi:
            preview = f"https://api.semanticscholar.org/DOI:{p.doi}"
            dl = p.pdf_url
        elif p.arxiv_id:
            preview = f"https://arxiv.org/abs/{p.arxiv_id}"
            dl = f"https://arxiv.org/pdf/{p.arxiv_id}.pdf"
        return preview, dl

    return {
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "abstract": (p.abstract or "")[:500],
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "is_oa": p.is_oa,
                "pdf_url": p.pdf_url,
                "preview_url": _paper_preview(p)[0],
                "download_url": _paper_preview(p)[1],
                "scihub_url": f"https://sci-hub.se/{p.doi}" if p.doi else None,
                "source": _paper_source(p),
                "doi_url": f"https://doi.org/{p.doi}" if p.doi else None,
                "cnki_url": _cnki_search_url(p.title) if p.title else None,
                "google_scholar_url": _gs_search_url(p.title) if p.title else None,
                "semantic_scholar_url": (
                    f"https://api.semanticscholar.org/CorpusID:{p.paper_id}"
                    if p.paper_id
                    else None
                ),
            }
            for p in papers
        ],
        "counts": {
            "total": len(papers),
            "oa_downloadable": sum(1 for p in papers if p.is_oa and p.pdf_url),
            "has_preview": sum(1 for p in papers if p.doi or p.arxiv_id),
        },
    }


def _paper_source(p) -> str:
    t = p.title or ""
    if t.startswith("[知网]"):
        return "cnki"
    if p.paper_id.startswith("gs:"):
        return "google_scholar"
    return "other"


def _cnki_search_url(title: str) -> str:
    import urllib.parse

    return f"https://kns.cnki.net/kns8/defaultresult/index?kwd={urllib.parse.quote(title[:50])}"


def _gs_search_url(title: str) -> str:
    import urllib.parse

    return f"https://scholar.google.com/scholar?q={urllib.parse.quote(title[:80])}"


@app.post("/download_bulk")
async def api_download_bulk(request: Request):
    """一键下载 OA 论文 → 自动解析 → 构建知识库（含图表）。接收 {urls, topic_id}"""
    import asyncio

    from fastapi.responses import Response

    req = await request.json()
    urls = req.get("urls", [])
    topic_id = req.get("topic_id", "bulk")
    if not urls:
        raise HTTPException(400, "No URLs provided")

    state = active_topics.get(topic_id)
    if state:
        state.step = "downloading"
        state.status = TopicStatus.BUILDING
        state.dl_total = len(urls)
        state.current = 0

    pdf_contents: list[tuple[str, bytes]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        sem = asyncio.Semaphore(3)

        async def _fetch(i, url):
            async with sem:
                try:
                    r = await client.get(url, follow_redirects=True)
                    if r.status_code == 200 and len(r.content) > 1000:
                        return (f"paper_{i+1}.pdf", r.content)
                except Exception:
                    pass
                return None

        results = await asyncio.gather(
            *(_fetch(i, url) for i, url in enumerate(urls[:50]))
        )
        pdf_contents = [r for r in results if r]

    # ---- 导入知识库 ----
    imported_papers = 0
    imported_chunks = 0
    imported_images = 0
    if state and pdf_contents:
        state.step = "indexing"
        papers: list[PaperMeta] = []
        fulltext_chunks: dict[str, list[str]] = {}
        all_images: list[dict] = []

        for fname, content in pdf_contents:
            try:
                result = parse_pdf(content, fname)
                paper_id = uuid.uuid4().hex[:16]
                papers.append(
                    PaperMeta(
                        paper_id=paper_id, title=fname.replace(".pdf", ""), is_oa=True
                    )
                )
                chunks = chunk_text(result["full_text"])
                if chunks:
                    fulltext_chunks[paper_id] = chunks
                for img in result["images"]:
                    img["paper_id"] = paper_id
                all_images.extend(result["images"])
            except Exception:
                pass

        # 构建文本层
        if papers:
            table_name = await build_knowledge_base(papers, fulltext_chunks, topic_id)
            imported_papers = len(papers)
            imported_chunks = sum(len(v) for v in fulltext_chunks.values())

        imported_images = len(all_images)

        state.lancedb_table = table_name
        state.uploaded_papers = imported_papers
        state.total_papers = imported_papers
        state.total_images = imported_images
        state.status = TopicStatus.READY
        save_topic_state(topic_id)
        state.step = "ready"

    # 打包 ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in pdf_contents:
            zf.writestr(fname, content)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=papers_{topic_id}.zip",
            "X-Imported": str(imported_papers),
            "X-Chunks": str(imported_chunks),
            "X-Images": str(imported_images),
        },
    )


@app.post("/upload_pdf/{topic_id}")
async def api_upload_pdf(topic_id: str, files: list[UploadFile]):
    state = active_topics.get(topic_id)
    if not state:
        raise HTTPException(404, "Topic not found")

    papers: list[PaperMeta] = []
    fulltext_chunks: dict[str, list[str]] = {}
    all_images: list[dict] = []

    for file in files:
        if not file.filename or not file.filename.endswith(".pdf"):
            continue
        pdf_bytes = await file.read()
        result = parse_pdf(pdf_bytes, file.filename)
        paper_id = uuid.uuid4().hex[:16]
        papers.append(
            PaperMeta(
                paper_id=paper_id, title=file.filename.replace(".pdf", ""), is_oa=True
            )
        )
        chunks = chunk_text(result["full_text"])
        if chunks:
            fulltext_chunks[paper_id] = chunks
        for img in result["images"]:
            img["paper_id"] = paper_id
        all_images.extend(result["images"])

    table_name = await build_knowledge_base(papers, fulltext_chunks, topic_id)

    if all_images:
        image_records = []
        for img in all_images:
            img_path = img["image_path"]
            image_records.append(
                {
                    "paper_id": img.get("paper_id", ""),
                    "title": Path(img_path).stem,
                    "text": "",
                    "image_path": img_path,
                    "page_number": img.get("page_num", 0),
                }
            )
        store_image_descriptions(table_name, image_records)

    state.lancedb_table = table_name
    state.uploaded_papers = len(papers)
    state.total_papers = len(papers)
    state.total_images = len(all_images)
    state.status = TopicStatus.READY
    save_topic_state(topic_id)
    state.step = "ready"

    return {
        "ok": True,
        "papers": len(papers),
        "chunks": sum(len(v) for v in fulltext_chunks.values()),
        "images": len(all_images),
    }


@app.get("/topic_status/{topic_id}")
async def api_topic_status(topic_id: str):
    state = active_topics.get(topic_id)
    if not state:
        raise HTTPException(404, "Topic not found")
    ready = state.status == TopicStatus.READY
    return {
        "status": "ready" if ready else "building",
        "progress": f"{state.uploaded_papers} 篇论文, {state.total_images} 张图片"
        if ready
        else state.step,
        "step": state.step,
        "current": state.current,
        "total": state.dl_total,
        "cn_papers": getattr(state, "cn_papers", 0),
        "en_papers": getattr(state, "en_papers", 0),
        "failed": getattr(state, "dl_failed", 0),
    }


@app.post("/chat")
async def api_chat(request: Request):
    """通用 AI 聊天：直接调 DeepSeek，无 RAG 检索。与知识库共享对话历史。"""
    req = await request.json()
    question = sanitize_input(req.get("question", ""))
    if not question:
        raise HTTPException(400, "需要 question 参数")
    from openai import AsyncOpenAI

    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    history = req.get("history", [])
    msgs = []
    for h in (history or [])[-8:]:
        role = "user" if h.get("r") == "u" else "assistant"
        msgs.append({"role": role, "content": str(h.get("c", ""))[:400]})
    msgs.append({"role": "user", "content": question})
    resp = await client.chat.completions.create(
        model=DEEPSEEK_MODEL, temperature=0.3, max_tokens=500, timeout=15, messages=msgs
    )
    return {"answer": resp.choices[0].message.content.strip()}


def _find_kb_table() -> str:
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    raw = db.list_tables()
    all_t = raw.tables if hasattr(raw, "tables") else []
    best, best_n = "", 0
    for t in all_t:
        try:
            n = db.open_table(t).count_rows()
        except:
            continue
        if n > best_n:
            best, best_n = t, n
    if not best or best_n <= 5:
        raise HTTPException(404, "No knowledge base available. Upload PDFs first.")
    return str(best)


@app.post("/ask")
async def api_ask(req: AskRequest):
    import lancedb
    from openai import AsyncOpenAI

    from config import (
        DEEPSEEK_API_KEY,
        DEEPSEEK_BASE_URL,
        DEEPSEEK_MODEL,
        LANCEDB_DIR,
        LLM_GEN_MAX_TOKENS,
        LLM_GEN_TEMPERATURE,
        LLM_GEN_TIMEOUT,
    )
    from knowledge_base import _cosine_search, _embed_texts_sync

    question = sanitize_input(req.question)
    state = active_topics.get(req.topic_id)
    table = state.lancedb_table if state and state.lancedb_table else ""
    if not table:
        table = _find_kb_table()

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    qv = _embed_texts_sync([question])[0]
    db = lancedb.connect(str(LANCEDB_DIR))
    res = _cosine_search(
        db.open_table(table), qv, 5, "is_fulltext = true AND is_image = false"
    )
    ctx = "\n".join([r.get("text", "")[:300] for r in (res or [])[:3]])

    history = req.history or []
    hist_text = ""
    for h in history[-4:]:
        role = "\u7528\u6237" if h.get("r") == "u" else "AI"
        hist_text += f"{role}: {str(h.get('c',''))[:200]}\n"

    prompt = f"\u57fa\u4e8e\u4e0a\u4e0b\u6587\u7b80\u6d01\u56de\u7b54\uff08100-300\u5b57\uff09\u3002\u5982\u679c\u4fe1\u606f\u4e0d\u8db3\u8bf7\u8bf4\u660e\u3002\n\u4e0a\u4e0b\u6587\uff1a{ctx}\n\u95ee\u9898\uff1a{question}\n\u56de\u7b54\uff1a"
    if hist_text:
        prompt = f"\u5bf9\u8bdd\u5386\u53f2\uff1a\n{hist_text}\n{prompt}"

    resp = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=LLM_GEN_TEMPERATURE,
        max_tokens=LLM_GEN_MAX_TOKENS,
        timeout=LLM_GEN_TIMEOUT,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": resp.choices[0].message.content.strip(),
        "references": [],
        "supplement": [],
        "images": [],
    }


@app.post("/agent/langgraph")
async def api_agent_langgraph(request: Request):
    """LangGraph Agent 端点（图状工作流，支持断点续跑和人工确认）。
    准备2 §334+§1104+§1251：StateGraph + 条件边 + 循环状态管理。"""
    req = await request.json()
    action = req.get("action", "run")  # run | resume
    task = req.get("task", "")
    thread_id = req.get("thread_id", "default")
    max_steps = req.get("max_steps", 10)

    if action == "resume":
        if not req.get("confirmed"):
            raise HTTPException(400, "resume 需要 confirmed=true")
        from agent.langgraph_workflow import resume_langgraph_agent

        result = await resume_langgraph_agent(thread_id, confirmed=True)
        return result

    if not task:
        raise HTTPException(400, "需要 task 参数")
    from agent.langgraph_workflow import run_langgraph_agent

    result = await run_langgraph_agent(task, max_steps=max_steps, thread_id=thread_id)
    return result


@app.post("/agent/watch")
async def api_agent_watch(request: Request):
    """Agent 监控面板 SSE：实时展示状态机流转、工具调用、Token 消耗。"""
    import json
    import time

    from fastapi.responses import StreamingResponse

    req = await request.json()
    task = req.get("task", "")
    thread_id = req.get("thread_id", f"watch_{uuid.uuid4().hex[:6]}")
    if not task:
        raise HTTPException(400, "需要 task 参数")

    async def stream():
        s = (
            lambda e, d: f"data: {json.dumps({'event': e, **d}, ensure_ascii=False)}\n\n"
        )
        t0 = time.time()
        tokens_total = 0

        yield s(
            "init", {"task": task, "thread": thread_id, "ts": f"{time.time()-t0:.1f}s"}
        )

        # 用 LangGraph 流式执行
        from agent.langgraph_workflow import get_agent_graph

        graph = get_agent_graph()
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "messages": [{"role": "user", "content": f"调研任务：{task}"}],
            "max_steps": 10,
            "steps": 0,
            "result": "",
            "confirmed": False,
            "need_confirm": None,
        }

        try:
            node_count = 0
            async for chunk in graph.astream(initial, config, stream_mode="updates"):
                node_count += 1
                elapsed = f"{time.time() - t0:.1f}s"
                for node_name, node_output in chunk.items():
                    # 估算 token
                    est = (
                        sum(len(str(v)) for v in node_output.values()) // 4
                        if isinstance(node_output, dict)
                        else len(str(node_output)) // 4
                    )
                    tokens_total += est

                    if node_name == "agent":
                        msgs = node_output.get("messages", [])
                        last = msgs[-1] if msgs else {}
                        has_tool = (
                            hasattr(last, "tool_calls") and last.tool_calls
                            if hasattr(last, "tool_calls")
                            else False
                        )
                        yield s(
                            "node_enter",
                            {
                                "node": "agent",
                                "step": node_count,
                                "tool_calls": bool(has_tool),
                                "tokens": tokens_total,
                                "elapsed": elapsed,
                            },
                        )
                    elif node_name == "tools":
                        yield s(
                            "node_enter",
                            {
                                "node": "tools",
                                "step": node_count,
                                "tokens": tokens_total,
                                "elapsed": elapsed,
                            },
                        )
                    elif node_name == "reflect":
                        result_text = str(node_output.get("result", ""))[:200]
                        yield s(
                            "node_enter",
                            {
                                "node": "reflect",
                                "step": node_count,
                                "tokens": tokens_total,
                                "elapsed": elapsed,
                                "result_preview": result_text,
                            },
                        )

                    yield s(
                        "node_done",
                        {
                            "node": node_name,
                            "step": node_count,
                            "tokens": tokens_total,
                            "elapsed": elapsed,
                        },
                    )

        except Exception as e:
            yield s("error", {"error": str(e), "elapsed": f"{time.time()-t0:.1f}s"})

        yield s(
            "done",
            {
                "steps": node_count,
                "tokens": tokens_total,
                "total_elapsed": f"{time.time()-t0:.1f}s",
            },
        )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/agent/multi")
async def api_agent_multi(request: Request):
    """Planner->Workers->Synthesizer."""
    req = await request.json()
    task = req.get("task", "")
    table = req.get("table", "")
    if not task:
        raise HTTPException(400, "需要 task 参数")
    from agent.langgraph_workflow import run_multi_agent

    return await run_multi_agent(task, table)


@app.get("/agent/report")
async def api_agent_report():
    """Agent评估报告：成功率/推理步数/工具准确率/影子测试."""
    from agent.langgraph_workflow import get_agent_evaluator

    return get_agent_evaluator().report()


@app.post("/ask/stream")
async def api_ask_stream(request: Request):
    """流式 RAG 问答：SSE 逐 token 返回，用户即时看到生成内容。"""
    from fastapi.responses import StreamingResponse

    req = await request.json()
    topic_id = req.get("topic_id", "")
    question = sanitize_input(req.get("question", ""))
    if not topic_id or not question:
        raise HTTPException(400, "需要 topic_id 和 question")

    state = active_topics.get(topic_id)
    if not state or not state.lancedb_table:
        raise HTTPException(404, "Topic not found or not ready")

    async def generate():
        from openai import AsyncOpenAI

        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        from knowledge_base import retrieve_with_images
        from qa_engine import _expand_query

        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        # Quick retrieval
        expanded = await _expand_query(question) if len(question) > 15 else question
        text_res, _, _ = retrieve_with_images(state.lancedb_table, expanded, top_k=5)
        ctx = "\n".join([r.get("text", "")[:500] for r in (text_res or list())[:3]])

        prompt = f"基于上下文简洁回答（100-200字）：\n{ctx}\n问题：{question}\n回答："
        import json

        try:
            stream = await client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/ask/trace")
async def api_ask_trace(request: Request):
    """RAG 流水线追踪端点：SSE 流式返回每步耗时和中间结果，供 pipeline.html 可视化。
    面试展示用——直观看到检索→改写→召回→重排→生成的完整链路。"""
    import json
    import time

    from fastapi.responses import StreamingResponse

    req = await request.json()
    question = sanitize_input(req.get("question", ""))
    topic_id = req.get("topic_id", "")
    if not question:
        raise HTTPException(400, "需要 question 参数")

    state = active_topics.get(topic_id) if topic_id else None
    kb_table = state.lancedb_table if state else None
    if not kb_table:
        # Auto-find any available KB
        import lancedb

        from config import LANCEDB_DIR

        db = lancedb.connect(str(LANCEDB_DIR))
        raw = db.list_tables()
        all_t = raw.tables if hasattr(raw, "tables") else []
        best, best_n = "", 0
        for t in all_t:
            try:
                n = db.open_table(t).count_rows()
                if n > best_n:
                    best, best_n = t, n
            except:
                pass
        kb_table = str(best) if best_n > 100 else None
    if not kb_table:
        raise HTTPException(404, "Topic 不存在或无可用知识库")

    async def trace():
        s = (
            lambda step, data: f"data: {json.dumps({'step': step, **data}, ensure_ascii=False)}\n\n"
        )

        t0 = time.time()

        # Step 1: 查询改写
        yield s("rewrite_start", {"question": question, "ts": f"{time.time()-t0:.2f}s"})
        from qa_engine import _expand_query, _is_complex_query

        expanded = question
        if len(question) >= 15 or any(
            kw in question for kw in ["对比", "区别", "优缺点", "机制"]
        ):
            t1 = time.time()
            expanded = await _expand_query(question)
            dt = time.time() - t1
            yield s(
                "rewrite_done",
                {
                    "original": question,
                    "expanded": expanded,
                    "changed": expanded != question,
                    "elapsed": f"{dt:.2f}s",
                },
            )
        else:
            yield s("rewrite_skip", {"reason": "短问题跳过改写", "elapsed": "0s"})

        # Step 2: 向量检索
        import lancedb
        import numpy as np

        from knowledge_base import LANCEDB_DIR, _cosine_search, _embed_texts_sync

        t1 = time.time()
        qv = _embed_texts_sync([expanded])[0]
        db = lancedb.connect(str(LANCEDB_DIR))
        tbl = db.open_table(kb_table)
        vec_res = _cosine_search(tbl, qv, 5, "is_fulltext = true AND is_image = false")
        dt_v = time.time() - t1
        top_score = float(
            np.dot(np.array(qv), np.array(vec_res[0].get("vector", qv)))
            if vec_res
            else 0
        )
        vec_titles = [r.get("title", "")[:40] for r in vec_res[:3]] if vec_res else []
        yield s(
            "vector_done",
            {
                "results": len(vec_res),
                "top_score": round(top_score, 4),
                "top_titles": vec_titles,
                "elapsed": f"{dt_v:.2f}s",
            },
        )

        # Step 3: BM25 全文检索
        t1 = time.time()
        try:
            bm25 = tbl.search(expanded, query_type="fts").limit(5).to_list()
            dt_b = time.time() - t1
            yield s(
                "bm25_done",
                {
                    "results": len(bm25),
                    "top_titles": [r.get("title", "")[:40] for r in bm25[:3]]
                    if bm25
                    else [],
                    "elapsed": f"{dt_b:.2f}s",
                },
            )
        except Exception:
            bm25 = []
            yield s(
                "bm25_done", {"results": 0, "elapsed": "N/A", "note": "LanceDB FTS 不可用"}
            )

        # Step 4: RRF 融合 + 自适应权重
        t1 = time.time()
        try:
            from hybrid_retriever import AdaptiveHybridRetriever

            hr = AdaptiveHybridRetriever(kb_table)
            alpha = hr._compute_adaptive_weight(expanded)
            text_res, _ = hr.search(
                expanded, top_k=5, filter_expr="is_fulltext = true AND is_image = false"
            )
            dt_f = time.time() - t1
            yield s(
                "fusion_done",
                {
                    "alpha_bm25": round(alpha, 3),
                    "alpha_vector": round(1 - alpha, 3),
                    "fused_count": len(text_res) if text_res else 0,
                    "strategy": "BM25偏向"
                    if alpha > 0.6
                    else ("均衡" if alpha > 0.4 else "语义偏向"),
                    "elapsed": f"{dt_f:.2f}s",
                },
            )
        except Exception:
            text_res = vec_res
            yield s(
                "fusion_done",
                {
                    "alpha_bm25": 0.5,
                    "fused_count": len(text_res),
                    "elapsed": "fallback",
                },
            )

        # Step 5: 级联重排
        t1 = time.time()
        from qa_engine import _rerank

        if _is_complex_query(question) and len(text_res) > 3:
            ranked = _rerank(expanded, text_res, top_k=5)
            dt_r = time.time() - t1
            yield s(
                "rerank_done",
                {
                    "before": len(text_res),
                    "after": len(ranked),
                    "reranked": True,
                    "top_titles": [r.get("title", "")[:40] for r in ranked[:5]]
                    if ranked
                    else [],
                    "elapsed": f"{dt_r:.2f}s",
                },
            )
            text_res = ranked
        else:
            yield s("rerank_skip", {"reason": "简单查询跳过重排"})

        # Step 6: 生成
        yield s("generate_start", {"elapsed": f"{time.time()-t0:.2f}s"})
        from openai import AsyncOpenAI

        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        ctx = "\n".join(
            [
                f"[{i+1}] {r.get('title','N/A')}\n{r.get('text','')[:300]}"
                for i, r in enumerate((text_res or list())[:3])
            ]
        )
        prompt = f"基于上下文回答（100-300字）。信息不足请说明。\n上下文：{ctx}\n问题：{question}\n回答："

        t_gen = time.time()
        try:
            stream = await client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=0.3,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            tokens = []
            async for chunk in stream:
                c = chunk.choices[0].delta.content
                if c:
                    tokens.append(c)
                    yield f"data: {json.dumps({'token': c}, ensure_ascii=False)}\n\n"
            answer = "".join(tokens)
            yield s(
                "generate_done",
                {
                    "tokens": len(tokens),
                    "elapsed": f"{time.time()-t_gen:.2f}s",
                    "total": f"{time.time()-t0:.2f}s",
                },
            )
        except Exception as e:
            yield s("generate_done", {"error": str(e), "elapsed": "fail"})

        yield s("done", {"total_elapsed": f"{time.time()-t0:.2f}s"})

    return StreamingResponse(trace(), media_type="text/event-stream")


@app.post("/agent/quick")
async def api_agent_quick(request: Request):
    """Agent 快速模式：直接用 KB 问答 + DeepSeek 总结，<10s。"""
    req = await request.json()
    question = sanitize_input(req.get("question", ""))
    from config import DEEPSEEK_MODEL

    model = req.get("model", "") or DEEPSEEK_MODEL
    if not question:
        raise HTTPException(400, "需要 question 参数")

    import lancedb

    from knowledge_base import LANCEDB_DIR, _embed_texts_sync

    # 找最大的 KB
    db = lancedb.connect(str(LANCEDB_DIR))
    raw = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    tables = (
        raw.tables
        if hasattr(raw, "tables")
        else (raw[0] if isinstance(raw, tuple) else raw)
    )
    # 找最大的知识库（按行数）
    best_kb, best_rows = "", 0
    for t in tables:
        try:
            n = db.open_table(t).count_rows()
            if n > best_rows:
                best_kb, best_rows = t, n
        except:
            pass
    kb = best_kb if best_rows > 5 else None
    if not kb:
        return {"result": "没有可用的知识库，请先构建知识库。"}

    # 直接检索（跳过改写，省 3s）
    qv = _embed_texts_sync([question])[0]
    from knowledge_base import _cosine_search

    res = _cosine_search(
        db.open_table(kb), qv, 5, "is_fulltext = true AND is_image = false"
    )
    ctx = "\n".join([r.get("text", "")[:500] for r in (res or list())[:3]])

    # 单次 DeepSeek 调用生成
    from openai import AsyncOpenAI

    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = f"基于上下文简洁回答（100-200字）。如果信息不足请说明。\n上下文：{ctx}\n问题：{question}\n回答："

    resp = await client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content.strip()

    return {"result": answer, "kb_used": kb}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=8001, limit_concurrency=20, timeout_keep_alive=30
    )
