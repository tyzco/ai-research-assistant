"""外部集成模块 — Zotero 导入 / Citation 链扩展 / 论文订阅。

Zotero Bridge:
  POST /zotero/import — 接收 Zotero metadata JSON, 自动创建课题+检索+建KB
  GET  /zotero/scan  — 扫描本地 Zotero SQLite 数据库 (macOS: ~/Zotero/zotero.sqlite)

Citation Chain (借鉴 ScholarFlow):
  POST /citation/chain — 从一篇论文出发，沿引用关系扩展上下游论文
  Semantic Scholar API: /paper/{id}/citations 和 /paper/{id}/references

Paper Subscription (借鉴 arxiv-tracker-agent):
  POST /subscribe  — 订阅研究方向，每天自动检索新论文
  GET  /subscribe/status — 查看订阅状态
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from models import PaperMeta

logger = logging.getLogger(__name__)

SUBSCRIPTION_FILE = Path("data/subscriptions.json")
SUBSCRIPTION_FILE.parent.mkdir(parents=True, exist_ok=True)

ZOTERO_DB_PATH = Path.home() / "Zotero" / "zotero.sqlite"
S2_CITATIONS_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
S2_REFERENCES_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
)
S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"


# ===== Zotero Bridge =====


async def import_from_zotero(metadata: dict, topic_id: str = "") -> dict:
    """从 Zotero metadata 导入论文并构建知识库。

    Zotero metadata 格式 (通过 Better BibTeX / Zotero API 导出):
    {
      "title": "...",
      "authors": [{"family": "Chen", "given": "T."}],
      "year": 2020,
      "DOI": "10.xxx/yyy",
      "abstract": "...",
      "collections": ["AI", "CV"]
    }
    """
    title = metadata.get("title", "")
    doi = metadata.get("DOI", "")
    authors_list = metadata.get("authors", [])
    authors = ", ".join(
        f"{a.get('family', '')} {a.get('given', '')}" for a in authors_list
    )
    year = (
        metadata.get("year")
        or metadata.get("issued", {}).get("date-parts", [[None]])[0][0]
    )
    abstract = metadata.get("abstract", "")[:1000]

    # 创建一个新的 topic 来存放这篇论文的 KB
    from topic_manager import create_topic

    topic_id = topic_id or f"zotero_{_safe_id(title)}"
    state = create_topic(title, topic_id=topic_id)

    # 构建 PaperMeta
    paper = PaperMeta(
        paper_id=topic_id,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        pdf_url=f"https://doi.org/{doi}" if doi else None,
        is_oa=False,
    )

    return {
        "ok": True,
        "topic_id": topic_id,
        "title": title,
        "doi": doi,
        "collections": metadata.get("collections", []),
        "tip": "论文已导入。可通过 /upload_pdf 上传 PDF 构建完整知识库。",
    }


async def scan_zotero_library(db_path: str = "") -> dict:
    """扫描本地 Zotero SQLite 数据库，列出可导入的论文。"""
    import sqlite3

    path = Path(db_path) if db_path else ZOTERO_DB_PATH
    if not path.exists():
        return {"ok": False, "error": f"Zotero 数据库未找到: {path}"}

    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row

        # Zotero SQLite schema: items, itemData, itemDataValues, fields, creators
        rows = conn.execute(
            """
            SELECT i.itemID, i.key as itemKey,
                   (SELECT v.value FROM itemData d
                    JOIN itemDataValues v ON d.valueID = v.valueID
                    JOIN fields f ON d.fieldID = f.fieldID
                    WHERE d.itemID = i.itemID AND f.fieldName = 'title'
                    LIMIT 1) as title,
                   (SELECT v.value FROM itemData d
                    JOIN itemDataValues v ON d.valueID = v.valueID
                    JOIN fields f ON d.fieldID = f.fieldID
                    WHERE d.itemID = i.itemID AND f.fieldName = 'DOI'
                    LIMIT 1) as doi,
                   (SELECT v.value FROM itemData d
                    JOIN itemDataValues v ON d.valueID = v.valueID
                    JOIN fields f ON d.fieldID = f.fieldID
                    WHERE d.itemID = i.itemID AND f.fieldName = 'date'
                    LIMIT 1) as year
            FROM items i
            WHERE i.itemTypeID = 2  -- journalArticle
            LIMIT 200
        """
        ).fetchall()

        conn.close()

        papers = []
        for r in rows:
            if r["title"]:
                papers.append(
                    {
                        "zotero_key": r["itemKey"],
                        "title": r["title"][:80],
                        "doi": r["doi"],
                        "year": r["year"][:4] if r["year"] else "",
                    }
                )

        return {"ok": True, "total": len(papers), "papers": papers[:20]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ===== Citation Chain (借鉴 ScholarFlow) =====


async def expand_citation_chain(
    paper_id: str, direction: str = "both", depth: int = 1, api_key: str = ""
) -> dict:
    """从一篇论文出发，沿引用关系扩展上下游。
    paper_id 可以是 DOI / arXiv ID / Semantic Scholar Corpus ID。

    direction: "citations" (谁引用了这篇), "references" (这篇引用了谁), "both"
    depth: 扩展层数 (1-3)
    """
    from config import SEMANTIC_SCHOLAR_API_KEY

    api_key = api_key or SEMANTIC_SCHOLAR_API_KEY

    # 先解析 paper_id 为 S2 corpus ID
    s2_id = await _resolve_s2_id(paper_id, api_key)
    if not s2_id:
        return {"ok": False, "error": f"无法解析 paper ID: {paper_id}"}

    result = {
        "seed": {"id": s2_id},
        "citations": [],
        "references": [],
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        headers = {"x-api-key": api_key} if api_key else {}

        # 获取源论文详情
        try:
            seed_resp = await client.get(
                S2_PAPER_URL.format(paper_id=s2_id),
                params={"fields": "title,authors,year,abstract,citationCount"},
                headers=headers,
            )
            if seed_resp.status_code == 200:
                result["seed"] = seed_resp.json()
        except Exception:
            pass

        # 获取引用链
        if direction in ("citations", "both"):
            try:
                cit_resp = await client.get(
                    S2_CITATIONS_URL.format(paper_id=s2_id),
                    params={
                        "fields": "title,authors,year,abstract,citationCount",
                        "limit": 20,
                    },
                    headers=headers,
                )
                if cit_resp.status_code == 200:
                    data = cit_resp.json().get("data", [])
                    result["citations"] = [
                        {
                            "paper_id": c.get("citingPaper", {}).get("paperId", ""),
                            "title": c.get("citingPaper", {}).get("title", ""),
                            "authors": ", ".join(
                                a.get("name", "")
                                for a in (c.get("citingPaper", {}).get("authors") or [])
                            ),
                            "year": c.get("citingPaper", {}).get("year"),
                            "citation_count": c.get("citingPaper", {}).get(
                                "citationCount", 0
                            ),
                            "is_influential": c.get("isInfluential", False),
                        }
                        for c in data[:20]
                    ]
            except Exception as e:
                logger.warning(f"Citation expansion failed: {e}")

        if direction in ("references", "both"):
            try:
                ref_resp = await client.get(
                    S2_REFERENCES_URL.format(paper_id=s2_id),
                    params={
                        "fields": "title,authors,year,abstract,citationCount",
                        "limit": 20,
                    },
                    headers=headers,
                )
                if ref_resp.status_code == 200:
                    data = ref_resp.json().get("data", [])
                    result["references"] = [
                        {
                            "paper_id": r.get("citedPaper", {}).get("paperId", ""),
                            "title": r.get("citedPaper", {}).get("title", ""),
                            "authors": ", ".join(
                                a.get("name", "")
                                for a in (r.get("citedPaper", {}).get("authors") or [])
                            ),
                            "year": r.get("citedPaper", {}).get("year"),
                            "citation_count": r.get("citedPaper", {}).get(
                                "citationCount", 0
                            ),
                            "is_influential": r.get("isInfluential", False),
                        }
                        for r in data[:20]
                    ]
            except Exception as e:
                logger.warning(f"Reference expansion failed: {e}")

    return {"ok": True, **result}


async def _resolve_s2_id(paper_id: str, api_key: str = "") -> Optional[str]:
    """将 DOI / arXiv ID 解析为 Semantic Scholar Corpus ID。"""
    if paper_id.startswith("CorpusID:") or len(paper_id) == 40:
        return paper_id.replace("CorpusID:", "")

    query = paper_id
    if paper_id.startswith("10."):
        query = f"DOI:{paper_id}"
    elif "arxiv" in paper_id.lower():
        query = f"ArXiv:{paper_id.split('/')[-1]}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 1, "fields": "paperId"},
                headers={"x-api-key": api_key} if api_key else {},
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    return data[0].get("paperId")
        except Exception:
            pass
    return None


# ===== Paper Subscription (借鉴 arxiv-tracker-agent) =====


def _load_subscriptions() -> dict:
    if SUBSCRIPTION_FILE.exists():
        try:
            return json.loads(SUBSCRIPTION_FILE.read_text())
        except Exception:
            pass
    return {"subscriptions": []}


def _save_subscriptions(data: dict):
    SUBSCRIPTION_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def subscribe_topic(
    query: str,
    keywords_en: list[str] = None,
    keywords_cn: list[str] = None,
    interval_hours: int = 24,
) -> dict:
    """订阅研究方向，设置自动检索频率。"""
    data = _load_subscriptions()
    # 幂等: 同 query 不重复订阅 (自动订阅时避免每建一次课题就加一条重复)
    for existing in data["subscriptions"]:
        if existing["query"] == query:
            return {"ok": True, "subscription": existing, "existing": True}
    sub = {
        "id": f"sub_{len(data['subscriptions']) + 1}",
        "query": query,
        "keywords_en": keywords_en or [],
        "keywords_cn": keywords_cn or [],
        "interval_hours": interval_hours,
        "created_at": datetime.now().isoformat(),
        "last_run": None,
        "total_papers_found": 0,
    }
    data["subscriptions"].append(sub)
    _save_subscriptions(data)
    return {"ok": True, "subscription": sub}


def list_subscriptions() -> list[dict]:
    return _load_subscriptions().get("subscriptions", [])


def delete_subscription(sub_id: str) -> bool:
    data = _load_subscriptions()
    before = len(data["subscriptions"])
    data["subscriptions"] = [s for s in data["subscriptions"] if s["id"] != sub_id]
    _save_subscriptions(data)
    return len(data["subscriptions"]) < before


async def run_subscription(sub_id: str) -> dict:
    """执行单次订阅检索。"""
    data = _load_subscriptions()
    sub = next((s for s in data["subscriptions"] if s["id"] == sub_id), None)
    if not sub:
        return {"ok": False, "error": "订阅不存在"}

    from search import search_papers_for_topic

    papers = await search_papers_for_topic(
        sub["query"],
        keywords_en=sub["keywords_en"] or None,
        keywords_cn=sub["keywords_cn"] or None,
    )

    sub["last_run"] = datetime.now().isoformat()
    sub["total_papers_found"] += len(papers)
    _save_subscriptions(data)

    return {
        "ok": True,
        "subscription_id": sub_id,
        "query": sub["query"],
        "new_papers": len(papers),
        "total_found": sub["total_papers_found"],
        "top_5": [
            {"title": p.title, "year": p.year, "is_oa": p.is_oa} for p in papers[:5]
        ],
    }


def _safe_id(title: str) -> str:
    import hashlib

    return hashlib.md5(title.encode()).hexdigest()[:12]
