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
from config import ENABLE_VISION, IMAGE_DIR, PROJECT_ROOT
from downloader import chunk_text, parse_pdf
from knowledge_base import build_knowledge_base, store_image_descriptions
from models import AskRequest, CreateTopicRequest, PaperMeta, TopicStatus
from monitor import metrics, sanitize_input
from qa_engine import ask_question
from search import generate_search_strategy, search_papers_for_topic
from topic_manager import active_topics, create_topic
from vision import describe_images

# 消息历史存储（按 topic_id）
message_store: dict[str, list[dict]] = {}

app = FastAPI(title="AI Research Assistant", version="0.3.0")


@app.on_event("startup")
async def startup_preload():
    """预热：提前加载嵌入和重排序模型，避免首次请求卡顿。"""
    import logging

    logger = logging.getLogger("startup")
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

        # 图片描述 + 存储
        if all_images:
            if ENABLE_VISION:
                descs = await describe_images([img["image_path"] for img in all_images])
            image_records = []
            for img in all_images:
                img_path = img["image_path"]
                if img_path in descs:
                    image_records.append(
                        {
                            "paper_id": img.get("paper_id", ""),
                            "title": Path(img_path).stem,
                            "text": descs[img_path],
                            "image_path": img_path,
                            "page_number": img.get("page_num", 0),
                        }
                    )
            store_image_descriptions(table_name, image_records)
            imported_images = len(image_records)

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

    described_count = 0
    if all_images:
        if ENABLE_VISION:
            descs = await describe_images([img["image_path"] for img in all_images])
        image_records = []
        for img in all_images:
            img_path = img["image_path"]
            if img_path in descs:
                image_records.append(
                    {
                        "paper_id": img.get("paper_id", ""),
                        "title": Path(img_path).stem,
                        "text": descs[img_path],
                        "image_path": img_path,
                        "page_number": img.get("page_num", 0),
                    }
                )
        store_image_descriptions(table_name, image_records)
        described_count = len(image_records)

    state.lancedb_table = table_name
    state.uploaded_papers = len(papers)
    state.total_papers = len(papers)
    state.total_images = described_count
    state.status = TopicStatus.READY
    state.step = "ready"

    return {
        "ok": True,
        "papers": len(papers),
        "chunks": sum(len(v) for v in fulltext_chunks.values()),
        "images": len(all_images),
        "described": described_count,
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


@app.post("/agent/run")
async def api_agent_run(request: Request):
    """Agent 端点：高层次调研任务 → 自动规划执行 → 返回报告。"""
    req = await request.json()
    task = req.get("task", "")
    topic_id = req.get("topic_id", f"agent_{uuid.uuid4().hex[:8]}")
    max_steps = req.get("max_steps", 5)
    if not task:
        raise HTTPException(400, "需要 task 参数")
    from agent.agent_core import ResearchAgent

    agent = ResearchAgent(topic_id=topic_id, max_steps=max_steps)
    result = await agent.run(task)
    return {
        "success": result.get("success", False),
        "result": result.get("result", "")[:5000],
        "steps": [
            {
                "step": s["step"],
                "thought": s.get("thought", ""),
                "tool_result": s.get("tool_result", ""),
            }
            for s in result.get("steps", [])
        ],
    }


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
        ctx = "\n".join([r.get("text", "")[:500] for r in (text_res or [])[:3]])

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
    ctx = "\n".join([r.get("text", "")[:500] for r in (res or [])[:3]])

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
