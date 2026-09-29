"""FastAPI 独立服务（端口 8001）：检索策略 + 论文搜索 + PDF 上传 + 问答 + ZIP 批量下载。"""

import io
import threading
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

from logger import setup_logging

setup_logging()

# 模块级 logger (修复: 原仅 startup 函数内局部定义 → 上传/删除/搜索等错误分支
# 调用 logger.warning 时 NameError → 坏 PDF 上传 500 崩溃而非优雅跳过)
import logging

logger = logging.getLogger(__name__)

from auth import create_token, get_current_user, login_user, register_user, require_user
from config import (
    CN_EN_TERMS,
    CORS_ORIGINS,
    DEEPSEEK_MODEL,
    DOWNLOAD_CONCURRENCY,
    DOWNLOAD_TIMEOUT,
    IMAGE_DIR,
    LANCEDB_DIR,
    LLM_GEN_MAX_TOKENS,
    LLM_GEN_TEMPERATURE,
    LLM_GEN_TIMEOUT,
    LLM_QUICK_MAX_TOKENS,
    LLM_QUICK_TEMPERATURE,
    LLM_TRANSLATE_MAX_TOKENS,
    LLM_TRANSLATE_TEMPERATURE,
    LLM_TRANSLATE_TIMEOUT,
    MAX_UPLOAD_BYTES,
    PROJECT_ROOT,
    RATE_LIMIT_REQUESTS_PER_SEC,
)
from downloader import chunk_text, parse_pdf
from knowledge_base import build_knowledge_base, store_image_descriptions
from models import AskRequest, CreateTopicRequest, PaperMeta, TopicStatus
from monitor import metrics, sanitize_input
from search import search_papers_for_topic
from topic_manager import active_topics, create_topic, save_topic_state

# 消息历史存储（按 topic_id）
_translation_lock = threading.Lock()  # DeepSeek 审查 #7: 翻译缓存并发写保护

app = FastAPI(title="AI Research Assistant", version="0.3.0")


@app.on_event("startup")
async def startup_preload():
    """预热：提前加载嵌入模型（非阻塞，失败不影响启动）。"""
    import asyncio
    import logging

    logger = logging.getLogger("startup")

    # 生产配置校验 (企业级: 启动即暴露缺失配置, 而非等用户请求才 500)
    import os

    from config import DEEPSEEK_API_KEY

    if not DEEPSEEK_API_KEY:
        logger.error(
            "DEEPSEEK_API_KEY 未配置! 所有 LLM 端点 (/chat /ask /agent) 将失败。"
            " 复制 .env.example 为 .env 并填入密钥。"
        )
    # JWT_SECRET_KEY 由 auth.py 直接从 env 读, 不在 config (修复: 原 from config import 崩启动)
    _jwt = os.getenv("JWT_SECRET_KEY", "")
    if "development-key" in _jwt or "change-me" in _jwt:
        logger.warning("JWT_SECRET_KEY 仍为默认值, 生产环境必须改为随机字符串 (否则可伪造 token)。")

    def _do_preload():
        try:
            from embedder import _get_embed_model

            _get_embed_model()
            logger.info("Embedding model preloaded")
        except Exception as e:
            logger.warning(f"Embed preload failed: {e}")
        try:
            from sentence_transformers import CrossEncoder

            from config import RERANK_MODEL

            CrossEncoder(RERANK_MODEL, max_length=512, local_files_only=True)
            logger.info("Reranker model preloaded")
        except Exception as e:
            logger.warning(f"Reranker preload failed: {e}")

    asyncio.get_event_loop().run_in_executor(None, _do_preload)

    # 启动后台订阅调度器 (每小时检查一次)
    try:
        from scheduler import start_scheduler

        start_scheduler(interval_seconds=3600)
    except Exception:
        pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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


def _list_lance_tables(db) -> list[str]:
    """统一 LanceDB 表列举 (DeepSeek 审查 #4: 兼容不同版本 API, 去重各处脆弱代码)。"""
    raw = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    return list(raw.tables) if hasattr(raw, "tables") else [str(t) for t in raw]


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # 请求 ID 追踪: 每次请求生成唯一 ID, 该请求内所有日志携带 (DeepSeek 建议)
    from logger import set_request_id

    set_request_id(uuid.uuid4().hex[:8])

    # DeepSeek 审查 #2: 反向代理下取 X-Forwarded-For 首个 IP, 否则用直连 IP
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    now = _time.time()

    # 定期清理不活跃 IP, 防止 _rate_limiter 内存无限增长
    if len(_rate_limiter) > 1024:
        stale = [k for k, v in _rate_limiter.items() if not v or now - v[-1] > 60.0]
        for k in stale:
            _rate_limiter.pop(k, None)

    window = [t for t in _rate_limiter[ip] if now - t < 1.0]
    if len(window) > RATE_LIMIT_REQUESTS_PER_SEC:
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "请求过于频繁，请稍后重试"}, status_code=429)
    window.append(now)
    _rate_limiter[ip] = window[-100:]  # trim old entries
    # 质量水位管控: 请求进出计数 → 检索层按并发水位动态升降质量档
    from load_gate import enter as _lg_enter, exit as _lg_exit

    _lg_enter()
    # /metrics 请求计数 + 延迟采样 (接线 MetricsTracker.record_request, 修复全零)
    _t0 = _time.time()
    try:
        resp = await call_next(request)
        # 请求日志 (结构化: IP 方法 路径 → 状态 耗时, 便于排障/演示)
        logger.info(
            f"[{ip}] {request.method} {request.url.path} → {resp.status_code} "
            f"({(_time.time() - _t0) * 1000:.0f}ms)"
        )
        return resp
    except Exception:
        logger.error(
            f"[{ip}] {request.method} {request.url.path} failed "
            f"({(_time.time() - _t0) * 1000:.0f}ms)"
        )
        raise
    finally:
        _lg_exit()  # 质量水位: 请求退出计数
        try:
            from monitor import metrics

            metrics.record_request(latency_ms=(_time.time() - _t0) * 1000)
        except Exception:
            pass


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器: 未捕获异常 → 统一 JSON 错误 (不暴露内部), 记录堆栈。"""
    from fastapi.responses import JSONResponse

    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试", "error_type": "internal"},
    )


async def _scope_gate(query: str, topic_mode: bool = False):
    """学术范围白名单: 非论文/学术问题 → 422 拒绝 (锁定工具用途, 防滥用)。

    异步内嵌, 嵌入层经线程池不阻塞事件循环。可被 SCOPE_GATE_ENABLED=0 关闭。
    topic_mode=True: 选题判断模式, 只拦硬非学术词 (裸选题短语天然是学术意图)。
    """
    from fastapi.responses import JSONResponse

    from config import SCOPE_GATE_ENABLED

    if not SCOPE_GATE_ENABLED:
        return None
    import asyncio

    from scope_gate import gate_query

    allowed, reason, level = await asyncio.to_thread(gate_query, query, topic_mode)
    if allowed:
        return None
    logger.info(f"scope_gate reject ({level}): {str(query)[:50]}")
    return JSONResponse(
        status_code=422,
        content={"detail": reason, "code": "SCOPE_REJECTED", "level": level},
    )


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """K8s readiness probe：验证 SQLite + LanceDB + DEEPSEEK_API_KEY 均可用。"""
    try:
        from database import get_db

        get_db()._ensure_tables()
    except Exception as e:
        raise HTTPException(503, f"database not ready: {e}")
    try:
        import lancedb

        from config import LANCEDB_DIR

        lancedb.connect(str(LANCEDB_DIR))
    except Exception as e:
        raise HTTPException(503, f"lancedb not ready: {e}")
    # LLM 密钥缺失 → 核心功能不可用, 判未就绪 (这是生产最常见的"忘了配"故障)
    from config import DEEPSEEK_API_KEY

    if not DEEPSEEK_API_KEY:
        raise HTTPException(503, "DEEPSEEK_API_KEY not configured")
    return {"status": "ready"}


@app.get("/metrics")
async def api_metrics():
    """系统监控指标。"""
    d = metrics.to_dict()
    try:
        from load_gate import stats as _lg_stats

        d["load_gate"] = _lg_stats()  # 质量水位: 在飞并发 / 峰值 / 档位
    except Exception:
        pass
    return d


@app.get("/kg_data")
async def api_kg_data(table: str = ""):
    """知识图谱数据端点。table 可以是 topic_id 或 LanceDB table 名。"""
    import lancedb

    from knowledge_graph import build_graph_from_kb

    db = lancedb.connect(str(LANCEDB_DIR))
    all_tables = _list_lance_tables(db)

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
        tables = _list_lance_tables(db)
        best, best_n = "", 0
        for t in tables:
            try:
                n = db.open_table(t).count_rows()
                if n > best_n:
                    best, best_n = t, n
            except Exception:
                pass
        table = str(best) if best else ""
    if not table:
        raise HTTPException(404, "无可用知识库")

    from knowledge_graph import build_graph_from_kb, extract_entities_from_abstracts

    # 获取论文元数据
    graph = build_graph_from_kb(table)
    if not graph["nodes"]:
        raise HTTPException(404, "知识库无数据")

    # LLM 实体抽取：从 LanceDB 直接查摘要（graph nodes 不含 abstract）
    import lancedb as _lancedb

    _db = _lancedb.connect(str(LANCEDB_DIR))
    _tbl = _db.open_table(table)
    _rows = _tbl.search().limit(200).to_list()
    _seen = set()
    abstracts = []
    for r in _rows:
        pid = r.get("paper_id", "")
        if pid and pid not in _seen:
            _seen.add(pid)
            abstracts.append(
                {
                    "paper_id": pid,
                    "title": r.get("title") or pid,
                    "abstract": r.get("abstract", ""),
                }
            )
            if len(abstracts) >= 20:
                break
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
async def api_list_topics(
    user_id: str = Depends(get_current_user), filter: str = "active"
):
    from database import get_db

    db = get_db()
    rows = db.list_topics(user_id if user_id is not None else None)
    # 默认 active: 隐藏已结题/中止/归档 — 防止未完成课题一直充斥上下文
    if filter != "all":
        rows = [r for r in rows if r.get("status") not in ("settled", "aborted", "archived")]
    return [
        {
            "topic_id": r["topic_id"],
            "query": r["query"],
            "status": r["status"],
            "papers": r.get("total_papers", 0),
        }
        for r in rows
    ]


@app.delete("/topic/{topic_id}")
async def api_delete_topic(
    topic_id: str, user_id: str = Depends(get_current_user), confirm: bool = False
):
    """删除课题 (不可逆: 级联删会话/记忆)。**后端强制二次确认** —— 前端有 confirm 弹窗,
    但直接调 API 可绕过; 缺 confirm=true 一律 400, 防误调用导致课题不可逆丢失。"""
    _assert_topic_access(topic_id, user_id)
    if not confirm:
        raise HTTPException(
            400,
            "删除不可逆（级联删除该课题的对话与记忆）。请带 confirm=true 重试。",
        )
    if topic_id not in active_topics:
        raise HTTPException(404, "Topic not found")
    del active_topics[topic_id]
    from topic_manager import _save_topics

    _save_topics()
    # 持久化清理 (修复: 原只删内存 active_topics, SQLite 行残留 → 课题删后"复活"):
    #   db.delete_topic: 删 topics 行 (FK 级联删 messages/memories)
    #   close_memory:    释放该课题的记忆内存缓存
    try:
        from database import get_db

        get_db().delete_topic(topic_id)
    except Exception as e:
        logger.warning(f"delete_topic sqlite cleanup failed: {e}")
    try:
        from conversation_memory import close_memory

        close_memory(topic_id)
    except Exception as e:
        logger.warning(f"close_memory cleanup failed: {e}")
    return {"ok": True}


@app.post("/topic/{topic_id}/state")
async def api_topic_state(
    topic_id: str, request: Request, user_id: str = Depends(get_current_user)
):
    """课题状态机流转 (结题/中止/归档等)。{status, reason}
    权限: 仅课题归属用户可流转。审计日志记录流转历史。"""
    _assert_topic_access(topic_id, user_id)
    req = await request.json()
    new_status = (req.get("status") or "").strip().lower()
    reason = sanitize_input(str(req.get("reason") or ""))[:200]
    if not new_status:
        raise HTTPException(400, "需要 status 参数 (settled/aborted/archived/ready 等)")
    from topic_manager import transition_topic

    ok, msg = transition_topic(topic_id, new_status, user_id, reason)
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg, "status": new_status}


@app.get("/export/{topic_id}")
async def api_export_topic(topic_id: str):
    from fastapi.responses import Response

    state = active_topics.get(topic_id)
    if not state:
        raise HTTPException(404, "Topic not found")
    lines = [f"# {state.query}", "", f"创建时间: {state.created_at}"]
    if state.search_strategy:
        # state.search_strategy 是 SearchStrategy **模型**(或历史数据里的 str), 从不是 dict —
        # 直接 .get() 会 AttributeError, 导致整个导出 500。统一归一成 dict 再用。
        s = state.search_strategy
        if hasattr(s, "model_dump"):
            s = s.model_dump()
        elif isinstance(s, str):
            import json as _json

            try:
                s = _json.loads(s)
            except Exception:
                s = {}
        if not isinstance(s, dict):
            s = {}
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
    from database import get_db

    db = get_db()
    msgs = db.get_messages(topic_id)
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
    import json
    import re
    from pathlib import Path

    import config

    if not any(19968 <= ord(c) <= 40869 for c in query):
        return ""

    # 翻译缓存检查 (DeepSeek 对话建议: 避免重复 LLM 调用)
    if config.TRANSLATION_CACHE_ENABLED:
        cache_path = Path(config.TRANSLATION_CACHE_FILE)
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
                if query in cache:
                    return cache[query]
            except Exception:
                pass

    try:
        from llm_client import chat_with_retry

        resp = await chat_with_retry(
            model=DEEPSEEK_MODEL,
            temperature=LLM_TRANSLATE_TEMPERATURE,
            max_tokens=LLM_TRANSLATE_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": f"将以下中文学术术语翻译为英文检索关键词，只返回英文逗号分隔，不要解释：{query}",
                }
            ],
            timeout=LLM_TRANSLATE_TIMEOUT,
        )
        text = (resp.choices[0].message.content or "").strip()
        result = re.sub(r"[^a-zA-Z0-9,\- ]", "", text)[:100]

        # 写入缓存
        if result and config.TRANSLATION_CACHE_ENABLED:
            _update_translation_cache(query, result)

        return result
    except Exception:
        import logging

        logging.getLogger("api").warning(
            "CN translation failed for query: %s", query[:50]
        )
        return ""


def _update_translation_cache(query: str, translation: str):
    """更新翻译缓存 JSON 文件 (LRU 淘汰旧条目, 最多 500 条)。
    DeepSeek 审查 #7: 加锁保护并发读写, 防止 JSON 文件损坏。"""
    import json
    from pathlib import Path

    import config

    with _translation_lock:
        cache_path = Path(config.TRANSLATION_CACHE_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:
                pass
        cache[query] = translation
        # 超出上限时淘汰最旧的条目 (FIFO)
        if len(cache) > config.TRANSLATION_CACHE_MAX_ENTRIES:
            oldest = list(cache.keys())[
                : len(cache) - config.TRANSLATION_CACHE_MAX_ENTRIES
            ]
            for k in oldest:
                del cache[k]
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def _quick_keywords(query: str) -> dict:
    import re

    import config

    words = [w.strip() for w in re.split(r"[，,、\s]+", query) if w.strip()]
    cn = [w for w in words if any(19968 <= ord(c) <= 40869 for c in w)]
    en = [w.lower() for w in words if w.isascii() and len(w) >= 2]

    # jieba 分词: 中文输入自动分词，组合为多词英文查询 (DeepSeek 对话建议)
    if config.USE_JIEBA_TOKENIZATION and cn:
        try:
            import jieba

            tokens = list(jieba.cut(query))
            cn = (
                list(dict.fromkeys([t.strip() for t in tokens if len(t.strip()) >= 2]))
                or cn
            )
            cn = cn[: config.CN_TOKENIZATION_TOP_K]
        except ImportError:
            pass  # jieba 不可用，用原分词结果

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
async def api_create_topic(
    req: CreateTopicRequest, user_id: str = Depends(get_current_user)
):
    """创建课题：本地关键词 + LLM翻译(非阻塞增强)。
    如果 topic 已存在（刷新页面恢复），保留已有策略和对话，不覆盖。"""
    query = sanitize_input(req.query)
    # topic_mode: create_topic 收的是研究方向 (裸选题名), 语义即学术, 只拦硬非学术词
    if (blk := await _scope_gate(query, topic_mode=True)) is not None:
        return blk
    # 写论文模式 (reuse_kb): 已有同名/相似且已建库的课题 → 直接复用, 跳过多源检索
    if req.reuse_kb:
        reused_tid = _find_kb_topic(query)
        if reused_tid:
            state = active_topics.get(reused_tid)
            strategy = (
                state.search_strategy
                if state and state.search_strategy
                else _quick_keywords(query)
            )
            return {"topic_id": reused_tid, "strategy": strategy, "reused": True}
        # 语义相似 → 确认式建议 (不自动复用; 高精度场景防错接库, 由用户决定)
        suggest = _find_semantic_similar_topic(query)
        if suggest:
            return {
                "suggest": {"topic_id": suggest[0], "query": suggest[1]},
                "reused": False,
            }
    is_new = not req.topic_id or req.topic_id not in active_topics
    state = create_topic(
        query, topic_id=req.topic_id if req.topic_id else "", user_id=user_id
    )

    if is_new or not state.search_strategy:
        strategy = _quick_keywords(query)
        state.search_strategy = strategy
    else:
        strategy = state.search_strategy

    # 后台LLM翻译增强英文关键词（非阻塞）——仅对新课题
    if is_new:

        async def _enhance():
            try:
                en_llm = await _translate_cn_query(query)
                if en_llm:
                    terms = [t.strip() for t in en_llm.split(",") if t.strip()]
                    if terms:
                        strategy["keywords_en"] = list(
                            set(terms + strategy["keywords_en"])
                        )
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

        # 主动推送 (蓝海核心): 自动订阅新课题, 后台每日检索新进展
        async def _auto_subscribe():
            try:
                from integrations import subscribe_topic

                subscribe_topic(
                    query,
                    keywords_en=strategy.get("keywords_en"),
                    keywords_cn=strategy.get("keywords_cn"),
                    user_id=user_id or "",
                )
            except Exception:
                pass

        asyncio.create_task(_auto_subscribe())
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
        }

    async def generate():
        s = (
            lambda e, d: f"data: {_json.dumps({'event': e, **d}, ensure_ascii=False)}\n\n"
        )
        t0 = time.time()

        # 完整检索 + 冷门拓宽: 保证展示/建库 ≥100 篇最相关论文 (可下载优先, top-100 截断)
        from search import fetch_relevant_papers

        papers = await fetch_relevant_papers(query, target=100)
        seen, uniq = set(), []
        for p in papers:
            if p.paper_id in seen:
                continue
            seen.add(p.paper_id)
            uniq.append(p)
        for i in range(0, len(uniq), 12):
            yield s(
                "batch",
                {
                    "source": "all",
                    "count": len(uniq),
                    "papers": [_fmt(p) for p in uniq[i : i + 12]],
                },
            )
        yield s("done", {"total": len(uniq), "elapsed": f"{time.time()-t0:.1f}s"})

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
async def api_download_bulk(request: Request, user_id: str = Depends(get_current_user)):
    """一键下载 OA 论文 → 自动解析 → 构建知识库（含图表）。接收 {urls, topic_id}"""
    import asyncio

    from fastapi.responses import Response

    req = await request.json()
    urls = req.get("urls", [])
    topic_id = req.get("topic_id", "bulk")
    _assert_topic_access(topic_id, user_id)
    if not urls:
        raise HTTPException(400, "No URLs provided")
    # 成本闸: 批量下载是"不可逆 + 高耗时"操作 (下载+解析+嵌入建库)。
    # 前端有按钮, 但直接调 API 会无提示地跑掉几分钟 → 未确认时先回报估算。
    from config import (
        COST_GATE_ENABLED,
        COST_GATE_MIN_PAPERS,
        COST_GATE_SEC_PER_PAPER,
    )

    if (
        COST_GATE_ENABLED
        and not req.get("confirm")
        and len(urls) >= COST_GATE_MIN_PAPERS
    ):
        _est_s = int(len(urls) * COST_GATE_SEC_PER_PAPER)
        return {
            "needs_confirm": True,
            "estimate": {
                "papers": len(urls),
                "est_seconds": _est_s,
                "est_minutes": round(_est_s / 60, 1),
                "note": (
                    f"将下载并解析 {len(urls)} 篇 PDF 并建库（本地嵌入, 无 API 费用; "
                    f"主要成本是时间, 约 {round(_est_s/60,1)} 分钟）。"
                    f"确认请带 confirm=true 重试。"
                ),
            },
        }

    state = active_topics.get(topic_id)
    if state:
        state.step = "downloading"
        state.status = TopicStatus.BUILDING
        state.progress.dl_total = len(urls)
        state.progress.current = 0

    pdf_contents: list[tuple[str, bytes]] = []
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
        sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)  # DeepSeek 审查 #14: 配置化

        async def _fetch(i, url):
            async with sem:
                try:
                    r = await client.get(url, follow_redirects=True)
                    if r.status_code == 200 and len(r.content) > 1000:
                        return (f"paper_{i+1}.pdf", r.content)
                except Exception:
                    pass
                # arxiv.org 被墙/不稳定 → 回退到 export.arxiv.org 镜像 (实测可达)
                if "arxiv.org/pdf/" in url and "export.arxiv.org" not in url:
                    alt = url.replace("arxiv.org/pdf/", "export.arxiv.org/pdf/")
                    try:
                        r2 = await client.get(alt, follow_redirects=True)
                        if r2.status_code == 200 and len(r2.content) > 1000:
                            return (f"paper_{i+1}.pdf", r2.content)
                    except Exception:
                        pass
                return None

        # 分批下载并逐批更新进度 (原 gather 并行一次性返回 → 进度卡 0/N 然后跳)
        _urls = urls[:500]
        _total = len(_urls)
        pdf_contents = []
        _B = 4  # 每批并发 4 篇
        for _start in range(0, _total, _B):
            _chunk_urls = _urls[_start : _start + _B]
            _idx0 = _start
            _rs = await asyncio.gather(
                *(_fetch(_idx0 + i, u) for i, u in enumerate(_chunk_urls))
            )
            for _r in _rs:
                if _r:
                    pdf_contents.append(_r)
            if state:
                state.progress.current = min(_idx0 + _B, _total)
                state.progress.dl_total = _total
        try:
            from monitor import metrics

            metrics.record_download()  # /metrics 下载计数 (接线 MetricsTracker.record_download)
        except Exception:
            pass

    # 如果全部下载失败，更新状态让前端停止轮询
    if not pdf_contents:
        if state:
            state.step = "ready"
            state.status = TopicStatus.READY
            state.error = f"下载失败：{len(urls)} 篇论文均无法获取（检查 OA 状态或网络连接）"
            save_topic_state(topic_id)
        raise HTTPException(400, "所有论文下载失败，请检查 OA 状态或网络连接")

    # ---- 导入知识库 ----
    imported_papers = 0
    imported_chunks = 0
    imported_images = 0
    if state:
        state.step = "indexing"
        state.progress.current = 0  # 重置, 索引阶段独立计数 (下载阶段已走到 N)
        papers: list[PaperMeta] = []
        fulltext_chunks: dict[str, list[str]] = {}
        all_images: list[dict] = []

        for fname, content in pdf_contents:
            try:
                state.progress.current += 1  # 进度: 逐篇解析/索引
                # DeepSeek 审查 #13: CPU 密集解析放入线程池
                result = await asyncio.to_thread(parse_pdf, content, fname)
                paper_id = uuid.uuid4().hex[:16]
                papers.append(
                    PaperMeta(
                        paper_id=paper_id, title=fname.replace(".pdf", ""), is_oa=True
                    )
                )
                chunks = await asyncio.to_thread(chunk_text, result["full_text"])
                if chunks:
                    fulltext_chunks[paper_id] = chunks
                for img in result["images"]:
                    img["paper_id"] = paper_id
                all_images.extend(result["images"])
            except Exception:
                pass

        # 构建文本层
        table_name = ""
        if papers:
            table_name = await build_knowledge_base(papers, fulltext_chunks, topic_id)
            imported_papers = len(papers)
            imported_chunks = sum(len(v) for v in fulltext_chunks.values())

        imported_images = len(all_images)

        state.lancedb_table = table_name
        state.stats.uploaded_papers = imported_papers
        state.stats.total_papers = imported_papers
        state.stats.total_images = imported_images
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
async def api_upload_pdf(
    topic_id: str, files: list[UploadFile], user_id: str = Depends(get_current_user)
):
    _assert_topic_access(topic_id, user_id)
    state = active_topics.get(topic_id)
    if not state:
        raise HTTPException(404, "Topic not found")

    papers: list[PaperMeta] = []
    fulltext_chunks: dict[str, list[str]] = {}
    all_images: list[dict] = []

    import asyncio

    for file in files:
        if not file.filename or not file.filename.endswith(".pdf"):
            continue
        pdf_bytes = await file.read()
        # DeepSeek 审查 #15: 大小上限 + PDF 魔数校验 (真 PDF 以 %PDF 开头)
        if len(pdf_bytes) > MAX_UPLOAD_BYTES:
            logger.warning(f"Upload skipped (too large): {file.filename}")
            continue
        if not pdf_bytes[:1024].lstrip().startswith(b"%PDF"):
            logger.warning(f"Upload skipped (not a valid PDF): {file.filename}")
            continue
        # DeepSeek 审查 #13: CPU 密集解析放入线程池, 不阻塞事件循环
        result = await asyncio.to_thread(parse_pdf, pdf_bytes, file.filename)
        paper_id = uuid.uuid4().hex[:16]
        papers.append(
            PaperMeta(
                paper_id=paper_id, title=file.filename.replace(".pdf", ""), is_oa=True
            )
        )
        chunks = await asyncio.to_thread(chunk_text, result["full_text"])
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
    state.stats.uploaded_papers = len(papers)
    state.stats.total_papers = len(papers)
    state.stats.total_images = len(all_images)
    state.status = TopicStatus.READY
    save_topic_state(topic_id)
    state.step = "ready"

    return {
        "ok": True,
        "papers": len(papers),
        "chunks": sum(len(v) for v in fulltext_chunks.values()),
        "images": len(all_images),
    }


@app.post("/upload/data")
async def api_upload_data(request: Request, user_id: str = Depends(get_current_user)):
    """上传数据文件 (CSV/XLSX/JSON/TXT) 供 read_csv 等工具读取。
    仅允许数据类扩展名, 存到 data_uploads/, 返回 filepath 供 agent 工具使用。"""
    import re as _re
    from pathlib import Path

    form = await request.form()
    file = form.get("file")
    if not file or not getattr(file, "filename", ""):
        raise HTTPException(400, "需要 file 字段")
    allowed = {".csv", ".xlsx", ".xls", ".json", ".txt", ".tsv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"仅支持数据文件: {sorted(allowed)}")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "文件过大 (限 10MB)")
    upload_dir = Path("data_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _re.sub(r"[^\w\-\.]", "_", Path(file.filename).name)
    dest = upload_dir / safe_name
    dest.write_bytes(content)
    return {
        "ok": True,
        "filepath": str(dest),
        "filename": safe_name,
        "size": len(content),
    }


@app.get("/topic_status/{topic_id}")
async def api_topic_status(topic_id: str, user_id: str = Depends(get_current_user)):
    _assert_topic_access(topic_id, user_id)
    state = active_topics.get(topic_id)
    if not state:
        raise HTTPException(404, "Topic not found")
    ready = state.status == TopicStatus.READY
    # DeepSeek 审查 #5: getattr 安全访问, 防 progress/stats 缺失时 500
    stats = state.stats
    progress = state.progress
    uploaded = getattr(stats, "uploaded_papers", 0) if stats else 0
    total_images = getattr(stats, "total_images", 0) if stats else 0
    current = getattr(progress, "current", 0) if progress else 0
    dl_total = getattr(progress, "dl_total", 0) if progress else 0
    cn_papers = getattr(progress, "cn_papers", 0) if progress else 0
    en_papers = getattr(progress, "en_papers", 0) if progress else 0
    dl_failed = getattr(progress, "dl_failed", 0) if progress else 0
    return {
        "status": "ready" if ready else "building",
        "progress": f"{uploaded} 篇论文, {total_images} 张图片" if ready else state.step,
        "step": state.step,
        "current": current,
        "total": dl_total,
        "cn_papers": cn_papers,
        "en_papers": en_papers,
        "failed": dl_failed,
    }


@app.post("/research_gap")
async def api_research_gap(request: Request, user_id: str = Depends(get_current_user)):
    """一键选题可行性判断 (蓝海钩子): SSE 流式逐字输出报告 (进度 + token)。"""
    import json as _json

    from fastapi.responses import StreamingResponse

    req = await request.json()
    topic = sanitize_input(req.get("topic", ""))
    if not topic:
        raise HTTPException(400, "需要 topic 参数")
    if (blk := await _scope_gate(topic, topic_mode=True)) is not None:
        return blk
    _quota_guard(request, user_id)
    from research_gap import stream_analyze

    async def gen():
        async for kind, payload in stream_analyze(topic):
            if kind == "status":
                yield f"data: {_json.dumps({'event': 'status', 'message': payload}, ensure_ascii=False)}\n\n"
            elif kind == "token":
                yield f"data: {_json.dumps({'event': 'token', 'token': payload}, ensure_ascii=False)}\n\n"
            elif kind == "done":
                yield f"data: {_json.dumps({'event': 'done', 'n_papers': payload}, ensure_ascii=False)}\n\n"
            else:  # error
                yield f"data: {_json.dumps({'event': 'error', 'message': payload}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/chat")
async def api_chat(request: Request):
    """通用 AI 聊天：直接调 DeepSeek，无 RAG 检索。与知识库共享对话历史。"""
    req = await request.json()
    question = sanitize_input(req.get("question", ""))
    if not question:
        raise HTTPException(400, "需要 question 参数")
    if (blk := await _scope_gate(question)) is not None:
        return blk
    _quota_guard(request)  # 按量付费闸门: 超额 → 402
    from config import DEEPSEEK_MODEL, MEMORY_HISTORY_TURNS
    from llm_client import chat_with_retry

    history = req.get("history", [])
    msgs = []
    for h in (history or [])[-MEMORY_HISTORY_TURNS:]:
        role = "user" if h.get("r") == "u" else "assistant"
        msgs.append({"role": role, "content": str(h.get("c", ""))[:400]})
    msgs.append({"role": "user", "content": question})

    resp = await chat_with_retry(
        model=DEEPSEEK_MODEL,
        temperature=LLM_GEN_TEMPERATURE,
        max_tokens=LLM_GEN_MAX_TOKENS,
        timeout=LLM_GEN_TIMEOUT,
        messages=msgs,
        extra_body={"thinking": {"type": "disabled"}},  # 推理模型禁用推理, 防输出空串
    )
    answer = resp.choices[0].message.content.strip()
    # 按量付费: 记录本次 LLM 用量
    _quota_record(
        request,
        "",
        (resp.usage.prompt_tokens if resp.usage else 0)
        + (resp.usage.completion_tokens if resp.usage else 0),
    )

    # DeepSeek 审查 #1: 保存消息移到 return 之前 (原代码在 return 后, 不可达),
    # 且改用 SQLite 持久化 (#12: 移除未使用的 message_store)。
    tid = req.get("topic_id")
    if tid:
        try:
            from database import get_db

            db = get_db()
            db.add_message(tid, "user", question)
            db.add_message(tid, "assistant", answer)
        except Exception as e:
            logger.warning(f"Failed to save chat message for {tid}: {e}")

    return {"answer": answer}


def _find_kb_topic(query: str):
    """写论文模式复用: 找已有同名/相似且已建库的课题, 复用跳过检索。"""
    from topic_manager import active_topics

    q_norm = (query or "").strip().lower()
    if not q_norm:
        return None

    def _bigrams(s: str) -> set:
        s = s.replace(" ", "")
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

    # 精确匹配优先
    for tid, state in active_topics.items():
        if state.lancedb_table and (state.query or "").strip().lower() == q_norm:
            return tid
    # 模糊: 子串包含 或 二元组重叠率≥0.5 (词序/措辞不同但核心相同, 如 "医学RAG" vs "RAG医学")
    b1 = _bigrams(q_norm)
    for tid, state in active_topics.items():
        if not state.lancedb_table:
            continue
        tq = (state.query or "").strip().lower()
        if not tq:
            continue
        if q_norm in tq or tq in q_norm:
            return tid
        b2 = _bigrams(tq)
        if b1 and b2 and len(b1 & b2) / min(len(b1), len(b2)) >= 0.5:
            return tid
    return None


def _find_semantic_similar_topic(query: str, threshold: float = 0.70):
    """语义相似课题检测 (复用建议, 非自动 — 高精度场景确认式才安全)。
    返回 (topic_id, query) 或 None。字符串精确匹配已在 _find_kb_topic 处理, 这里跳过。"""
    from topic_manager import active_topics

    q = (query or "").strip().lower()
    if not q:
        return None
    try:
        import numpy as np

        from embedder import embed_texts
    except Exception:
        return None
    # 候选课题查询词 (有库 + 非空 + 非精确同词 — 精确已由 _find_kb_topic 处理)
    cands = [
        (tid, st.query.strip().lower())
        for tid, st in active_topics.items()
        if st.lancedb_table and st.query and st.query.strip().lower() != q
    ]
    if not cands:
        return None

    def _bigrams(s: str) -> set:
        s = s.replace(" ", "")
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

    # 双重门槛: 语义相似 ≥ threshold 且 共享 ≥1 个关键词二元组 (同一核心方向才建议)。
    # bge 绝对相似度膨胀 (脑机接口 vs 图神经网络 也 >0.7), 单靠阈值会误配跨方向课题。
    b1 = _bigrams(q)
    qv = np.array(embed_texts([query])[0])
    pvs = np.asarray(embed_texts([c for _, c in cands]))
    sims = pvs @ qv
    for i, (tid, cq) in enumerate(cands):
        if sims[i] >= threshold and (b1 & _bigrams(cq)):
            return (tid, cq)
    return None


def _caller_identity(request: Request, user_id: str | None) -> str:
    """调用者身份: 登录用户用 user_id; 匿名用客户端 IP。

    与 quota 的「匿名按 IP 记账」同一口径 —— 用于给 agent run 绑定发起者,
    使人工确认闸只接受发起者本人的批准 (此前任何拿到 run_id 的人都能批准)。
    """
    if user_id:
        return f"user:{user_id}"
    fwd = request.headers.get("x-forwarded-for", "") if request else ""
    ip = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request and request.client else ""
    )
    return f"ip:{ip}"


def _assert_topic_access(topic_id: str, user_id: str | None = None) -> None:
    """多用户课题访问校验 (微观层面补全隔离):
    用户专属课题(user_id 非空)仅归属者可访问; 匿名/共享课题(NULL)人人可访问。
    与 /topics 列表级过滤构成双保险, 防止凭 topic_id 直接越权读他人课题。"""
    try:
        from database import get_db

        row = get_db().get_topic(topic_id)
    except Exception:
        return  # 数据库异常不阻断, 交由端点 404
    if row is None:
        return  # topic 不存在, 端点自行 404
    owner = row.get("user_id")
    if owner and owner != user_id:
        raise HTTPException(403, "该课题属于其他用户，无权访问")


def _resolve_kb_table(topic_id: str) -> str:
    """为指定 topic 解析正确的 LanceDB 表名。优先级：
    1. active_topics 内存中的 lancedb_table
    2. SQLite 数据库持久化的映射
    3. 都不存在则报错
    """
    state = active_topics.get(topic_id)
    if state and state.lancedb_table:
        return state.lancedb_table
    from database import get_db

    db = get_db()
    row = db.get_topic(topic_id)
    if row and row.get("lancedb_table"):
        return row["lancedb_table"]
    raise HTTPException(
        404,
        f"Topic {topic_id} 尚未构建知识库。请先上传 PDF 或搜索论文后一键下载构建。",
    )


@app.post("/ask")
async def api_ask(req: AskRequest, user_id: str = Depends(get_current_user)):
    """RAG 问答：委托 qa_engine.ask_question 完整链路——
    指代消解 → 查询改写(校验) → 自适应混合检索 → 级联重排 → 父子分块扩展 → 三层 Prompt 生成。"""
    _assert_topic_access(req.topic_id, user_id)
    _quota_guard(request, user_id)  # 按量付费闸门: 超额 → 402
    from qa_engine import ask_question

    question = sanitize_input(req.question)
    if (blk := await _scope_gate(question)) is not None:
        return blk
    table = _resolve_kb_table(req.topic_id)

    result = await ask_question(
        table, question, model=req.model or "", memory_key=req.topic_id
    )
    # 按量付费: 估算本次用量 (prompt 中英字符 + 输出)
    _quota_record(request, user_id, int(len(question) / 1.5 + len(result.answer) / 1.5))
    # 记录消息到 SQLite 供 /export 导出
    from database import get_db

    get_db().add_message(req.topic_id, "user", question)
    get_db().add_message(req.topic_id, "assistant", result.answer)
    return {
        "answer": result.answer,
        "references": result.references,
        "supplement": result.supplement,
        "images": result.images,
    }


@app.post("/feedback")
async def api_feedback(request: Request):
    """用户对回答的质量反馈（数据飞轮）。"""
    req = await request.json()
    question = sanitize_input(req.get("question", ""))
    answer = req.get("answer", "")
    rating = req.get("rating", 0)  # 1=好, -1=差
    comment = req.get("comment", "")
    if not question or rating == 0:
        raise HTTPException(400, "需要 question 和 rating 参数")
    user_id = sanitize_input(str(req.get("user_id", "") or ""))
    from database import get_db

    get_db().record_feedback(question, answer, rating, comment, user_id=user_id)
    return {"ok": True, "total": get_db().get_feedback_stats()["total"]}


@app.get("/feedback/stats")
async def api_feedback_stats():
    """反馈统计数据。"""
    from database import get_db

    return get_db().get_feedback_stats()


# ===== 已合并: LangGraph 全部并入 pi_agent =====
# 原 /agent/langgraph (断点续跑) + /agent/watch (SSE 监控) + /agent/multi
# (Planner-Workers-Synthesizer) + /agent/report (评估器) 已全部移除:
# - ReAct 循环 / 人工确认门 / 实时 trace → pi_agent (run_id + /agent/pi/approve + trace.steps)
# - 多 agent 调研 → research_team (13 agent + 三审批闸门)
# 前端统一走 /agent/pi。


@app.post("/ask/stream")
async def api_ask_stream(request: Request):
    """流式 RAG 问答：走 ask_question 完整检索链路，流式返回生成 token。"""
    from fastapi.responses import StreamingResponse

    req = await request.json()
    topic_id = req.get("topic_id", "")
    question = sanitize_input(req.get("question", ""))
    if not topic_id or not question:
        raise HTTPException(400, "需要 topic_id 和 question")
    if (blk := await _scope_gate(question)) is not None:
        return blk

    table = _resolve_kb_table(topic_id)

    async def generate():
        import json

        from openai import AsyncOpenAI

        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        from qa_engine import _expand_query

        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        # 复用 qa_engine 的完整检索链路
        try:
            from hybrid_retriever import AdaptiveHybridRetriever

            # 查询改写
            expanded = (
                await _expand_query(question) if len(question) >= 15 else question
            )
            hr = AdaptiveHybridRetriever(table)
            text_res, _ = hr.search(
                expanded, top_k=5, filter_expr="is_fulltext = true AND is_image = false"
            )
        except Exception:
            from knowledge_base import retrieve_with_images

            text_res, _, _ = retrieve_with_images(table, question, top_k=5)

        ctx = "\n".join(
            f"[{i+1}] {r.get('title', 'N/A')}\n{r.get('text', '')[:2000]}"
            for i, r in enumerate((text_res or [])[:3])
        )
        prompt = f"基于上下文回答（100-300字）。信息不足请说明。\n上下文：{ctx}\n问题：{question}\n回答："

        try:
            stream = await client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_GEN_TEMPERATURE,
                max_tokens=LLM_GEN_MAX_TOKENS,
                stream=True,
                extra_body={"thinking": {"type": "disabled"}},
            )
            full_answer: list[str] = []
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.exception("ask stream failed: %s", e)
            yield f"data: {json.dumps({'error': '生成失败，请稍后重试'})}\n\n"

        # DeepSeek 审查 #3: 保存完整答案, 原代码 add_turn(question, "") 污染记忆
        try:
            from conversation_memory import get_memory

            get_memory(topic_id).add_turn(question, "".join(full_answer))
        except Exception:
            pass

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
    if (blk := await _scope_gate(question)) is not None:
        return blk

    state = active_topics.get(topic_id) if topic_id else None
    kb_table = state.lancedb_table if state else None
    if not kb_table:
        # Auto-find any available KB
        import lancedb

        from config import LANCEDB_DIR

        db = lancedb.connect(str(LANCEDB_DIR))
        all_t = _list_lance_tables(db)
        best, best_n = "", 0
        for t in all_t:
            try:
                n = db.open_table(t).count_rows()
                if n > best_n:
                    best, best_n = t, n
            except Exception:
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
                temperature=LLM_GEN_TEMPERATURE,
                max_tokens=LLM_GEN_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body={"thinking": {"type": "disabled"}},
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
            logger.exception("ask trace generation failed: %s", e)
            yield s("generate_done", {"error": "生成失败，请稍后重试", "elapsed": "fail"})

        yield s("done", {"total_elapsed": f"{time.time()-t0:.2f}s"})

    return StreamingResponse(trace(), media_type="text/event-stream")


@app.post("/agent/quick")
async def api_agent_quick(request: Request):
    """Agent 快速模式：直接用 KB 问答 + DeepSeek 总结，<10s。"""
    req = await request.json()
    question = sanitize_input(req.get("question", ""))
    _quota_guard(request)  # 按量付费闸门: 超额 → 402
    from config import DEEPSEEK_MODEL

    model = req.get("model", "") or DEEPSEEK_MODEL
    if not question:
        raise HTTPException(400, "需要 question 参数")
    if (blk := await _scope_gate(question)) is not None:
        return blk

    import lancedb

    from knowledge_base import LANCEDB_DIR, _embed_texts_sync

    # 找最大的 KB
    db = lancedb.connect(str(LANCEDB_DIR))
    tables = _list_lance_tables(db)
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
    from llm_client import chat_with_retry

    prompt = f"基于上下文简洁回答（100-200字）。如果信息不足请说明。\n上下文：{ctx}\n问题：{question}\n回答："

    resp = await chat_with_retry(
        model=model,
        temperature=LLM_QUICK_TEMPERATURE,
        max_tokens=LLM_QUICK_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content.strip()
    # 按量付费: 记录本次 LLM 用量
    _quota_record(
        request,
        "",
        (resp.usage.prompt_tokens if resp.usage else 0)
        + (resp.usage.completion_tokens if resp.usage else 0),
    )

    return {"result": answer, "kb_used": kb}


# ===== Pi-Agent 端点 (v3.0 Hook 驱动 + 上下文工程 + 工具管理) =====


@app.post("/agent/pi")
async def api_agent_pi(request: Request, user_id: str | None = Depends(get_current_user)):
    """Pi-Agent 风格主循环: trace→turn 生命周期 + 10 个 hook 节点 + 上下文压缩。
    危险工具 (need_confirm) 会暂停并返回 run_id, 由 /agent/pi/approve 确认后续跑。

    匿名可用 (与全站匿名优先一致), 但 run 会绑定发起者身份, 确认闸只认发起者。
    """
    req = await request.json()
    task = sanitize_input(req.get("task", ""))
    if not task:
        raise HTTPException(400, "需要 task 参数")
    if (blk := await _scope_gate(task)) is not None:
        return blk
    _quota_guard(request)  # 按量付费闸门: 超额 → 402

    max_steps = req.get("max_steps", 20)
    max_time = req.get("max_time", 300)
    goal = req.get("goal", "")
    skills = req.get("skills") or None  # 显式技能名数组, None=auto 意图选择
    # 之前漏读: 下面 get_agent() 传了 topic_id=topic_id, 而这里从未定义它 →
    # NameError → /agent/pi 每次调用都 500 (非流式端点一直是坏的;
    # /agent/pi/stream 读了这个字段所以前端没暴露出来)。
    topic_id = req.get("topic_id", "")

    from agent.pi_agent import get_agent

    agent = get_agent(
        max_steps=max_steps,
        max_time_seconds=max_time,
        goal=goal,
        enable_tool_check=True,
        enable_context_compaction=True,
        skills=skills,
        topic_id=topic_id,
        owner=_caller_identity(request, user_id),
    )
    result = await agent.run(task)
    result["run_id"] = agent.run_id
    # 按量付费: 记录本次 Agent 用量 (trace.total_tokens)
    _quota_record(request, "", result.get("trace", {}).get("total_tokens", 0))
    return result


@app.post("/agent/pi/stream")
async def api_agent_pi_stream(request: Request, user_id: str | None = Depends(get_current_user)):
    """Agent 流式执行: SSE 推送每步工具调用事件 + 最终结果。
    人工确认门暂停时推送 paused 事件 (run_id), 前端用 /agent/pi/approve 续跑。"""
    import asyncio
    import json as _json

    from fastapi.responses import StreamingResponse

    req = await request.json()
    task = sanitize_input(req.get("task", ""))
    if not task:
        raise HTTPException(400, "需要 task 参数")
    if (blk := await _scope_gate(task)) is not None:
        return blk
    max_steps = req.get("max_steps", 20)
    max_time = req.get("max_time", 300)
    goal = req.get("goal", "")
    skills = req.get("skills") or None
    topic_id = req.get("topic_id", "")

    # 当前课题知识库上下文: 让 agent 只查当前课题的表 (修复跨全部测试库乱答+慢)
    kb_context = ""
    if topic_id:
        try:
            from topic_manager import active_topics

            _st = active_topics.get(topic_id)
            if _st and getattr(_st, "lancedb_table", None):
                kb_context = (
                    f"当前课题「{_st.query}」的知识库表是 {_st.lancedb_table}。\n"
                    f"【强制】当前课题已有知识库，优先直接基于该库检索问答。\n"
                    f"① 禁止先调用 generate_search_strategy / search_papers 跑外部搜索；"
                    f"只有当库内检索确实无相关内容时，才可补充外部搜索。\n"
                    f"② 不要调用 search_knowledge_bases 去搜索其他表。\n"
                    f"③ 直接调用 agent_ask(table_name=\"{_st.lancedb_table}\", question=...) 或 ask_knowledge_base 获取库内内容。"
                )
        except Exception:
            pass

    from agent.pi_agent import get_agent

    queue = asyncio.Queue()

    async def step_cb(ev):
        await queue.put(ev)

    # get_agent("") 创建并注册新实例 → resume (approve) 可续跑
    agent = get_agent(
        "",
        max_steps=max_steps,
        max_time_seconds=max_time,
        goal=goal,
        skills=skills,
        step_cb=step_cb,
        kb_context=kb_context,
        topic_id=topic_id,
        owner=_caller_identity(request, user_id),
    )

    async def generate():
        run_task = asyncio.ensure_future(agent.run(task))
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                if run_task.done():
                    break
        result = await run_task
        result["run_id"] = agent.run_id
        yield (
            f"data: {_json.dumps({'event': 'result', **result}, ensure_ascii=False, default=str)}\n\n"
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/agent/pi/approve")
async def api_agent_pi_approve(request: Request, user_id: str | None = Depends(get_current_user)):
    """人工确认 pi-agent 暂停的危险工具, 恢复执行。
    POST {run_id, confirmed: true|false}

    只接受**发起该 run 的同一身份** (登录 user_id, 匿名则客户端 IP)。
    此前无任何校验: 任何人只要拿到 run_id 就能批准 agent 暂停的危险动作,
    等于人工确认闸形同虚设。
    """
    req = await request.json()
    run_id = req.get("run_id", "")
    if not run_id:
        raise HTTPException(400, "需要 run_id 参数")
    confirmed = bool(req.get("confirmed", True))
    from agent.pi_agent import get_agent

    try:
        agent = get_agent(run_id)
    except KeyError:
        raise HTTPException(404, f"run 不存在或已过期: {run_id}")
    caller = _caller_identity(request, user_id)
    if getattr(agent, "owner", "") and agent.owner != caller:
        raise HTTPException(403, "该 run 属于其他发起者，无权确认")
    result = await agent.resume(confirmed)
    result["run_id"] = run_id
    return result


@app.get("/agent/pi/stats")
async def api_agent_pi_stats(user_id: str = Depends(require_user)):
    """Pi-Agent 运行时统计: hooks 调用次数 + 上下文压缩次数 + 工具错误统计。

    需登录: 这是内部运行指标(hook 计数/上下文压缩/工具调用), 不是买家功能;
    此前裸露在无鉴权接口上。前端不调用此端点, 加鉴权无副作用。
    """
    from agent.context_engine import get_context_engine
    from agent.hooks import get_hooks
    from agent.tool_manager import get_tool_manager

    return {
        "hooks": get_hooks().get_stats(),
        "context": get_context_engine().get_stats(),
        "tools": get_tool_manager().get_stats(),
    }


@app.get("/agent/skills")
async def api_agent_skills():
    """获取所有可用技能及状态。"""
    from agent.skills import get_skills

    return {"skills": get_skills().list_all()}


@app.get("/suggest")
async def api_suggest(topic_id: str = "", user_id: str = Depends(get_current_user)):
    """阶段感知主动建议: 预判用户在「选题→初稿」旅程的位置, 递上下一步。
    这是"知用户、预判下一步"的智能路由第一版。"""
    from suggestions import rank_candidates
    from topic_manager import active_topics

    state = active_topics.get(topic_id) if topic_id else None
    if not state:
        return {
            "stage": "idle",
            "suggest": "输入研究方向，先做研究空白分析，确认这个选题有没有人做过",
            "action": "new_topic",
            "alternatives": [],
        }

    has_kb = bool(state.lancedb_table)
    has_papers = int(getattr(state.stats, "total_papers", 0) or 0) > 0
    has_msgs = 0
    try:
        from database import get_db

        has_msgs = len(get_db().get_messages(topic_id))
    except Exception:
        pass

    if not has_kb and not has_papers:
        stage, candidates = "created", [
            {"action": "search", "suggest": "查看论文卡片，看有没有相关文献"},
            {"action": "build_kb", "suggest": "直接下载 OA 论文建库"},
        ]
    elif not has_kb:
        stage, candidates = "searched", [
            {"action": "build_kb", "suggest": "下载 OA 全文构建知识库"},
            {"action": "ask", "suggest": "先基于摘要问几个问题"},
        ]
    elif has_msgs < 2:
        stage, candidates = "kb_ready", [
            {"action": "ask", "suggest": "直接提问，基于知识库回答"},
            {"action": "write", "suggest": "让 Agent 写综述/初稿"},
        ]
    else:
        stage, candidates = "writing", [
            {"action": "write", "suggest": "让 Agent 写初稿，写完做质量清单验收"},
            {"action": "ask", "suggest": "继续追问细节"},
        ]

    ranked = rank_candidates(stage, candidates)
    top = ranked[0]
    return {
        "stage": stage,
        "suggest": top["suggest"],
        "action": top["action"],
        "alternatives": [
            {"action": c["action"], "suggest": c["suggest"]} for c in ranked[1:]
        ],
    }


@app.post("/suggest/accept")
async def api_suggest_accept(request: Request):
    """记录用户接受了某阶段建议 — 主动路由的自我进化数据。"""
    from suggestions import record_accept

    req = await request.json()
    stage, action = req.get("stage", ""), req.get("action", "")
    if stage and action:
        record_accept(stage, action)
    return {"ok": True}


@app.get("/suggest/stats")
async def api_suggest_stats():
    """主动路由学习统计: 各阶段建议被接受次数 (越接受越优先推)。"""
    from suggestions import get_stats

    return {"stats": get_stats()}


@app.get("/agent/tools")
async def api_agent_tools():
    """列出所有已注册工具 + 被哪些技能引用 (含 tools_extra 插件动态加载)。"""
    from agent.skills import get_skills
    from agent.tool_registry import get_tools_schema

    skill_map: dict[str, list[str]] = {}
    for sk in get_skills().list_all():
        for t in sk.get("tools", []):
            skill_map.setdefault(t, []).append(sk["name"])

    tools = []
    for s in get_tools_schema():
        name = s["function"]["name"]
        tools.append(
            {
                "name": name,
                "description": s["function"]["description"][:120],
                "required": s["function"]["parameters"].get("required", []),
                "used_by_skills": skill_map.get(name, []),
            }
        )
    return {"total": len(tools), "tools": tools}


@app.post("/agent/tools/reload")
async def api_agent_tools_reload(user_id: str = Depends(require_user)):
    """重新扫描 tools_extra/ 插件目录 — 新增插件文件无需重启即注册生效。"""
    from agent.tool_registry import _load_plugin_tools, get_tools_schema

    added = _load_plugin_tools()
    return {"ok": True, "added": added, "total": len(get_tools_schema())}


# ===== 按量付费配额 (免费额度 → 用量追踪 → 充值) =====


def _quota_key(request: Request, user_id: str = "") -> str:
    """配额记账键: 登录按 user_id, 匿名按 IP。"""
    from quota import get_user_key

    ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    return get_user_key(user_id=user_id, ip=ip)


def _quota_guard(request: Request, user_id: str = "") -> str:
    """配额闸门: 超额 → 402; 否则返回记账键。"""
    from quota import check_quota

    key = _quota_key(request, user_id)
    if check_quota(key)["blocked"]:
        raise HTTPException(402, "免费额度已用完，请充值后继续使用")
    return key


def _quota_record(request: Request, user_id: str, tokens: int) -> None:
    try:
        from quota import record_usage

        record_usage(_quota_key(request, user_id), int(tokens or 0))
    except Exception:
        pass


# DeepSeek 账户余额缓存 (60s TTL, 避免每次刷新都打 DeepSeek)
_deepseek_balance_cache: dict = {"value": None, "ts": 0.0}


def _deepseek_balance() -> float | None:
    """查 DeepSeek 账户余额 (元)。失败/未配置返回 None。60s 缓存。"""
    import time

    import httpx

    from config import DEEPSEEK_API_KEY

    now = time.time()
    if _deepseek_balance_cache["value"] is not None and now - _deepseek_balance_cache["ts"] < 60:
        return _deepseek_balance_cache["value"]
    if not DEEPSEEK_API_KEY:
        return None
    try:
        r = httpx.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            timeout=6,
        )
        total = 0.0
        for b in r.json().get("balance_infos", []):
            total += float(b.get("total_balance", 0) or 0)
        _deepseek_balance_cache["value"] = round(total, 2)
        _deepseek_balance_cache["ts"] = now
        return _deepseek_balance_cache["value"]
    except Exception:
        return None


@app.get("/quota")
async def api_quota(request: Request, user_id: str = Depends(get_current_user)):
    """当前配额状态 (免费额度 / 已用 / 剩余) + DeepSeek 账户余额 (缓存 60s)。"""
    from quota import check_quota

    q = check_quota(_quota_key(request, user_id))
    return {"user_key": _quota_key(request, user_id), **q, "deepseek_balance": _deepseek_balance()}


@app.post("/quota/topup")
async def api_quota_topup(request: Request, user_id: str = Depends(get_current_user)):
    """充值占位: 增加可用额度。真实支付 (Stripe/支付宝) 接入后, 由支付回调调用此逻辑。"""
    from quota import top_up

    req = await request.json()
    tokens = int(req.get("tokens", 0))
    if tokens <= 0:
        raise HTTPException(400, "tokens 需为正整数")
    q = top_up(_quota_key(request, user_id), tokens)
    return {"ok": True, "top_up_tokens": tokens, **q}


@app.post("/quota/checkout")
async def api_quota_checkout(
    request: Request, user_id: str = Depends(get_current_user)
):
    """创建 Stripe Checkout 会话 — 充值 tokens → 跳转 Stripe 托管支付页。"""
    import stripe

    from config import STRIPE_PRICE_PER_1000_TOKENS, STRIPE_SECRET_KEY

    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "支付未配置 (STRIPE_SECRET_KEY)")
    stripe.api_key = STRIPE_SECRET_KEY
    req = await request.json()
    tokens = int(req.get("tokens", 0))
    if tokens <= 0:
        raise HTTPException(400, "tokens 需为正整数")
    amount_cents = int(tokens / 1000.0 * STRIPE_PRICE_PER_1000_TOKENS * 100)
    # Stripe 各币种最低金额不同 (USD $0.50 / HKD $4.00≈$0.51USD); 统一抬到 $1.00 兜底,
    # 否则 HKD 账户下 $0.50 换算后低于 HKD 下限 → 建单失败 (实测)
    amount_cents = max(100, amount_cents)
    user_key = _quota_key(request, user_id)
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": f"PaperLink LLM 额度 {tokens} tokens"},
                        "unit_amount": amount_cents,
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url="http://localhost:8001/quota/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="http://localhost:8001/quota/cancel",
            metadata={"user_key": user_key, "tokens": str(tokens)},
        )
        return {
            "ok": True,
            "checkout_url": session.url,
            "session_id": session.id,
            "tokens": tokens,
            "amount_cents": amount_cents,
        }
    except Exception as e:
        raise HTTPException(500, f"Stripe 会话创建失败: {e}")


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Stripe webhook: 支付成功 (checkout.session.completed) → 按 metadata 给用户充值。"""
    import stripe

    from config import STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    if not STRIPE_WEBHOOK_SECRET:
        return {"ok": False, "error": "STRIPE_WEBHOOK_SECRET 未配置 (用 stripe listen 获取)"}
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Webhook 签名校验失败: {e}")
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        meta = session.get("metadata", {}) or {}
        user_key = meta.get("user_key", "")
        tokens = int(meta.get("tokens", 0) or 0)
        if user_key and tokens:
            from quota import top_up

            top_up(user_key, tokens)
            logger.info(f"Stripe payment credited: +{tokens} tokens → {user_key}")
            return {"ok": True, "credited": tokens, "to": user_key}
    return {"ok": True, "received": event["type"]}


@app.get("/eval/results")
async def api_eval_results():
    """读取评估产物 (data/*.json) 聚合返回 — 供开发者中心可视化。"""
    import json as _json
    from pathlib import Path

    reports = {}
    for name in [
        "eval_result",
        "rag_standards_report",
        "rag_faithfulness_report",
        "embedding_benchmark",
        "e2e_report",
    ]:
        p = Path("data") / f"{name}.json"
        if p.exists():
            try:
                reports[name] = _json.loads(p.read_text())
            except Exception:
                reports[name] = {"error": "解析失败"}
    return {"reports": reports, "hint": "运行 run_eval.py / evaluate_rag_standards.py 生成"}


@app.get("/quota/success")
async def quota_success():
    return {"ok": True, "message": "充值成功，额度已到账"}


@app.get("/quota/cancel")
async def quota_cancel():
    return {"ok": True, "message": "充值已取消"}


@app.post("/agent/skills/toggle")
async def api_agent_skills_toggle(request: Request, user_id: str = Depends(require_user)):
    """启用/禁用技能。POST {name: "literature_review", enabled: true}"""
    req = await request.json()
    name = req.get("name", "")
    enabled = req.get("enabled", True)
    from agent.skills import get_skills

    skills = get_skills()
    if enabled:
        skills.enable(name)
    else:
        skills.disable(name)
    return {"ok": True, "active": sorted(skills.active)}


# ===== 虚拟科研团队 (13 Agent + 双轨审批) =====


@app.post("/team/start")
async def api_team_start(request: Request):
    """启动虚拟科研团队执行完整研究流程。
    每次 start 创建独立 run_id (多用户隔离)。返回含 run_id, 前端 /team/approve 必须携带。
    返回: {status, run_id, agent, name} 或 {status, run_id, result, history}"""
    req = await request.json()
    task = sanitize_input(req.get("task", ""))
    if not task:
        raise HTTPException(400, "需要 task 参数")
    if (blk := await _scope_gate(task)) is not None:
        return blk
    # 成本闸: 科研团队是 13 个 agent 的流水线 (多次 LLM 调用, 真金白银)。
    # 未确认时先回报估算 —— 防误点/误调用直接开跑烧钱。
    from config import COST_GATE_ENABLED, COST_GATE_TEAM_ESTIMATE

    if COST_GATE_ENABLED and not req.get("confirm"):
        return {
            "needs_confirm": True,
            "estimate": {
                "llm_calls": COST_GATE_TEAM_ESTIMATE,
                "note": (
                    f"科研团队将执行 {COST_GATE_TEAM_ESTIMATE} 个 agent 阶段"
                    f"（约 {COST_GATE_TEAM_ESTIMATE} 次 LLM 调用，含 3 道人工审批门）。"
                    f"确认请带 confirm=true 重试。"
                ),
            },
        }
    from agent.research_team import get_team

    team = get_team()  # 每次 start 独立实例, 不再全局单例互踩
    topic_id = sanitize_input(req.get("topic_id", ""))
    result = await team.execute(task, topic_id=topic_id)
    result["run_id"] = team.run_id
    return result


@app.post("/team/approve")
async def api_team_approve(request: Request):
    """人工确认审批闸门，继续执行后续流程。
    POST {run_id, feedback: "修正意见" (可选)}"""
    req = await request.json()
    run_id = req.get("run_id", "")
    if not run_id:
        raise HTTPException(400, "需要 run_id 参数")
    feedback = req.get("feedback", "")
    from agent.research_team import get_team

    try:
        team = get_team(run_id)
    except KeyError:
        raise HTTPException(404, f"run 不存在或已过期: {run_id}")
    result = await team.approve_and_continue(feedback)
    result["run_id"] = run_id
    return result


@app.get("/team/status")
async def api_team_status(run_id: str = ""):
    """查看指定 run 的当前状态和审批历史。GET /team/status?run_id=..."""
    if not run_id:
        raise HTTPException(400, "需要 run_id 查询参数")
    from agent.research_team import get_team

    try:
        team = get_team(run_id)
    except KeyError:
        raise HTTPException(404, f"run 不存在: {run_id}")
    return {
        "run_id": run_id,
        "current_stage": team.current_stage,
        "pending_approval": (
            {
                "gate": team.pending_approval.gate.value,
                "title": team.pending_approval.title,
                "summary": team.pending_approval.summary[:300],
            }
            if team.pending_approval
            else None
        ),
        "history": team.history,
        "completed_count": len(team.history),
    }


# ===== 论文导出 (DOCX / LaTeX / BibTeX / Markdown) =====


@app.post("/export/paper")
async def api_export_paper(request: Request):
    """导出论文为指定格式 (借鉴 OpenDraft 多格式导出)。
    POST {title, sections, references, format: "docx"|"latex"|"markdown"|"bibtex"}"""
    req = await request.json()
    title = req.get("title", "论文标题")
    sections = req.get("sections", {})
    references = req.get("references", [])
    fmt = req.get("format", "markdown")

    from exporter import export_bibtex, export_docx, export_latex, export_markdown

    exporters = {
        "markdown": export_markdown,
        "docx": export_docx,
        "latex": export_latex,
        "bibtex": export_bibtex,
    }
    if fmt not in exporters:
        raise HTTPException(400, f"不支持的格式: {fmt}，可选: {list(exporters.keys())}")

    filepath = exporters[fmt](title, sections, references)
    return {"ok": True, "format": fmt, "filepath": filepath}


@app.get("/file/download")
async def api_export_download(filepath: str = ""):
    """下载导出的论文文件 (仅限 data/exports/ 目录, 防路径穿越)。
    路径用 /file/download 而非 /export/download: 避免被先注册的 GET /export/{topic_id} 路由遮蔽。"""
    from pathlib import Path

    from exporter import EXPORT_DIR

    if not filepath:
        raise HTTPException(400, "缺少 filepath")
    if Path(filepath).is_absolute():
        raise HTTPException(400, "非法路径")
    root = EXPORT_DIR.resolve()
    target = Path(filepath).resolve()  # 相对 CWD 解析 (前端传 data/exports/xxx)
    # 穿越防护: 目标必须位于导出目录内 (resolve 已去 .. / 符号链接逃逸)
    if target != root and root not in target.parents:
        raise HTTPException(403, "路径不在导出目录内")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "文件不存在，请先调用 /export/paper 生成")
    return FileResponse(
        target, filename=target.name, media_type="application/octet-stream"
    )


# ===== Zotero Bridge (借鉴 LitScribe + 生态衔接) =====


@app.post("/zotero/import")
async def api_zotero_import(request: Request):
    """Zotero 论文导入：接收 Zotero metadata JSON，自动创建课题。
    可通过 Zotero Plugin 或 Better BibTeX 导出 JSON 调用。"""
    req = await request.json()
    from integrations import import_from_zotero

    result = await import_from_zotero(req, req.get("topic_id", ""))
    return result


@app.get("/zotero/scan")
async def api_zotero_scan():
    """扫描本地 Zotero 数据库 (macOS: ~/Zotero/zotero.sqlite)。"""
    from integrations import scan_zotero_library

    return await scan_zotero_library()


# ===== Citation Chain (借鉴 ScholarFlow 引文链扩展) =====


@app.post("/citation/chain")
async def api_citation_chain(request: Request):
    """引文链扩展：从一篇论文出发，沿引用关系发现上下游论文。
    POST {paper_id: "DOI or arXiv or S2 ID", direction: "citations|references|both", depth: 1}
    """
    req = await request.json()
    paper_id = req.get("paper_id", "")
    direction = req.get("direction", "both")
    depth = req.get("depth", 1)
    if not paper_id:
        raise HTTPException(400, "需要 paper_id 参数")
    from integrations import expand_citation_chain

    return await expand_citation_chain(paper_id, direction, depth)


# ===== Paper Subscription (借鉴 arxiv-tracker-agent) =====


@app.post("/subscribe")
async def api_subscribe(request: Request):
    """订阅研究方向，每天自动检索新论文。"""
    req = await request.json()
    query = sanitize_input(req.get("query", ""))
    if not query:
        raise HTTPException(400, "需要 query 参数")
    # topic_mode: subscribe 订阅的是研究方向, 语义即学术
    if (blk := await _scope_gate(query, topic_mode=True)) is not None:
        return blk
    from integrations import subscribe_topic

    return subscribe_topic(
        query,
        keywords_en=req.get("keywords_en"),
        keywords_cn=req.get("keywords_cn"),
        interval_hours=req.get("interval_hours", 24),
    )


@app.get("/subscribe")
async def api_list_subscriptions(user_id: str = Depends(get_current_user)):
    """查看当前用户的订阅 (多租户: 只返回归属自己的)。"""
    from integrations import list_subscriptions

    subs = [
        s
        for s in list_subscriptions()
        if (s.get("user_id") or "") == (user_id or "")
    ]
    return {"subscriptions": subs}


@app.get("/updates")
async def api_updates(user_id: str = Depends(get_current_user)):
    """主动推送 (蓝海核心): 返回当前用户每个订阅方向「最近一次」检索到的相关论文数。

    多租户隔离: 只返回归属当前用户的订阅 (user_id 匹配)。
    按订阅去重: 每个方向只显示最新一次检索 (避免天天跑的老订阅刷屏, 挤掉用户新订阅)。
    措辞诚实: 数字是「检索到的相关论文数」(实时 API 快照), 非严格增量「新增论文」。
    """
    import json
    from pathlib import Path

    log_file = Path("data/scheduler_log.json")
    if not log_file.exists():
        return {"updates": [], "total_new_papers": 0}
    try:
        entries = json.loads(log_file.read_text())
    except Exception:
        return {"updates": [], "total_new_papers": 0}
    mine = [
        e
        for e in entries
        if (e.get("new_papers") or 0) > 0 and e.get("user_id", "") == (user_id or "")
    ]
    mine.sort(key=lambda e: e.get("time", ""), reverse=True)
    # 按订阅去重: 每个 subscription_id 只保留最新一条
    latest: dict = {}
    for e in mine:
        key = e.get("subscription_id") or e.get("query") or "?"
        if key not in latest:
            latest[key] = e
    updates = sorted(latest.values(), key=lambda e: e.get("time", ""), reverse=True)
    return {
        "updates": updates[:10],
        "total_new_papers": sum(e.get("new_papers", 0) for e in updates),
        "note": "数字为各方向最近一次检索到的相关论文数（实时抓取公开学术库），非严格新增量。",
    }


@app.delete("/subscribe/{sub_id}")
async def api_delete_subscription(sub_id: str, user_id: str = Depends(get_current_user)):
    """删除订阅 (多租户: 只允许删除归属自己的订阅)。"""
    from integrations import _load_subscriptions, _save_subscriptions

    data = _load_subscriptions()
    subs = data.get("subscriptions", [])
    target = next((s for s in subs if s["id"] == sub_id), None)
    if target is None:
        return {"ok": False}
    if (target.get("user_id") or "") != (user_id or ""):
        raise HTTPException(403, "无权删除该订阅")
    data["subscriptions"] = [s for s in subs if s["id"] != sub_id]
    _save_subscriptions(data)
    return {"ok": True}


@app.post("/subscribe/{sub_id}/run")
async def api_run_subscription(sub_id: str):
    """手动触发一次订阅检索。"""
    from integrations import run_subscription

    return await run_subscription(sub_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=8001, limit_concurrency=20, timeout_keep_alive=30
    )
