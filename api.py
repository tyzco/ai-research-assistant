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
from config import IMAGE_DIR, PROJECT_ROOT
from downloader import chunk_text, parse_pdf
from knowledge_base import build_knowledge_base, store_image_descriptions
from models import AskRequest, CreateTopicRequest, PaperMeta, TopicStatus
from monitor import metrics, sanitize_input
from qa_engine import ask_question
from search import generate_search_strategy, search_papers_for_topic
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    """知识图谱数据端点（D3.js 力导向图使用）。"""
    from knowledge_graph import build_graph_from_kb

    if table:
        return build_graph_from_kb(table)
    # 找最大 KB
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        raw = db.list_tables()
        tables = (
            list(raw.tables)
            if hasattr(raw, "tables")
            else (list(raw[0]) if isinstance(raw, tuple) else list(raw))
        )
    except Exception:
        tables = []
    best, best_n = "", 0
    for t in tables:
        try:
            n = db.open_table(t).count_rows()
            if n > best_n:
                best, best_n = t, n
        except Exception:
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


@app.post("/register")
async def api_register(request: Request):
    req = await request.json()
    ok, msg = register_user(
        req.get("username", ""), req.get("password", ""), req.get("email", "")
    )
    if not ok:
        raise HTTPException(400, msg)
    token = create_token(req["username"])
    return {"access_token": token, "token_type": "bearer", "user": req["username"]}


@app.post("/login")
async def api_login(request: Request):
    req = await request.json()
    token = login_user(req.get("username", ""), req.get("password", ""))
    if not token:
        raise HTTPException(401, "用户名或密码错误")
    return {"access_token": token, "token_type": "bearer", "user": req["username"]}


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


@app.post("/create_topic")
async def api_create_topic(req: CreateTopicRequest):
    state = create_topic(sanitize_input(req.query))
    strategy = await generate_search_strategy(req.query)
    state.search_strategy = strategy
    message_store[state.topic_id] = [
        {"role": "system", "content": f"研究方向: {req.query}"}
    ]
    return {"topic_id": state.topic_id, "strategy": strategy}


@app.post("/search_papers")
async def api_search_papers(request: Request):
    """搜索论文。接收 {query, keywords_en (可选)}，传 keywords 可跳过重复 LLM 调用提速。"""
    req = await request.json()
    query = req.get("query", "")
    keywords_en = req.get("keywords_en")
    keywords_cn = req.get("keywords_cn")
    papers = await search_papers_for_topic(query, keywords_en, keywords_cn)
    return {
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "abstract": (p.abstract or "")[:300],
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "is_oa": p.is_oa,
                "pdf_url": p.pdf_url,
                "source": _paper_source(p),
                "doi_url": f"https://doi.org/{p.doi}" if p.doi else None,
                "cnki_url": _cnki_search_url(p.title) if not p.is_oa else None,
                "google_scholar_url": _gs_search_url(p.title) if not p.is_oa else None,
                "semantic_scholar_url": f"https://api.semanticscholar.org/CorpusID:{p.paper_id}"
                if p.paper_id
                else None,
            }
            for p in papers
        ]
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


@app.post("/ask")
async def api_ask(req: AskRequest):
    state = active_topics.get(req.topic_id)
    if not state or not state.lancedb_table:
        raise HTTPException(404, "Topic not found or not ready")
    result = await ask_question(
        state.lancedb_table, sanitize_input(req.question), model=req.model
    )
    msgs = message_store.setdefault(req.topic_id, [])
    msgs.append({"role": "user", "content": req.question})
    msgs.append({"role": "assistant", "content": result.answer})
    return result.model_dump()


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
                max_tokens=600,
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
    if not state or not state.lancedb_table:
        raise HTTPException(404, "Topic 不存在或知识库未就绪")

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
        tbl = db.open_table(state.lancedb_table)
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

            hr = AdaptiveHybridRetriever(state.lancedb_table)
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
                f"[{i+1}] {r.get('title','N/A')}\n{r.get('text','')[:800]}"
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

    prompt = f"基于上下文简洁回答（150-300字）。如果信息不足请说明。\n上下文：{ctx}\n问题：{question}\n回答："

    resp = await client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content.strip()

    return {"result": answer, "kb_used": kb}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
