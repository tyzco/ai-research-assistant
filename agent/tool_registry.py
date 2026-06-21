"""Agent 工具注册表：将现有功能封装为标准 Tool 接口（Function Calling 规范）。

每个工具声明：name, description, parameters(JSON Schema), execute 函数。
"""

import inspect
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---- 工具定义 ----

ToolFunc = Callable[..., Any]


class Tool:
    """LLM 可调用的工具。"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: ToolFunc,
        need_confirm: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema
        self.func = func
        self.need_confirm = need_confirm  # 是否需要人工确认

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, **kwargs) -> str:
        try:
            result = (
                await self.func(**kwargs)
                if inspect.iscoroutinefunction(self.func)
                else self.func(**kwargs)
            )
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)[:3000]
        except Exception as e:
            return f"工具执行失败: {e}"


# ---- 工具实现 ----


async def _tool_search_papers(
    query: str, keywords_en: list = None, keywords_cn: list = None
) -> dict:
    from search import search_papers_for_topic

    papers = await search_papers_for_topic(query, keywords_en, keywords_cn)
    return {
        "total": len(papers),
        "oa_count": sum(1 for p in papers if p.is_oa and p.pdf_url),
        "top_5": [
            {"title": p.title, "year": p.year, "is_oa": p.is_oa and bool(p.pdf_url)}
            for p in papers[:5]
        ],
    }


async def _tool_download_index(urls: list, topic_id: str) -> dict:
    import uuid

    import httpx

    from downloader import chunk_text, parse_pdf
    from knowledge_base import build_knowledge_base
    from models import PaperMeta

    pdf_contents = []
    async with httpx.AsyncClient(timeout=30) as client:
        for url in urls[:5]:
            try:
                r = await client.get(url, follow_redirects=True)
                if r.status_code == 200 and len(r.content) > 1000:
                    pdf_contents.append(
                        (f"paper_{uuid.uuid4().hex[:8]}.pdf", r.content)
                    )
            except Exception:
                pass

    papers, chunks = [], {}
    for fname, content in pdf_contents:
        result = parse_pdf(content, fname)
        pid = uuid.uuid4().hex[:12]
        papers.append(PaperMeta(paper_id=pid, title=fname, is_oa=True))
        c = chunk_text(result["full_text"])
        if c:
            chunks[pid] = c

    if papers:
        table = await build_knowledge_base(papers, chunks, topic_id)
        return {
            "ok": True,
            "papers": len(papers),
            "chunks": sum(len(v) for v in chunks.values()),
            "table": table,
        }
    return {"ok": False, "error": "没有成功下载任何论文"}


async def _tool_ask_kb(table_name: str, question: str) -> dict:
    from qa_engine import ask_question

    result = await ask_question(table_name, question)
    return {
        "answer": result.answer[:1500],
        "references": len(result.references),
        "supplement": len(result.supplement),
    }


async def _tool_generate_strategy(query: str) -> dict:
    from search import generate_search_strategy

    s = await generate_search_strategy(query)
    return {
        "keywords_cn": s.get("keywords_cn", [])[:5],
        "keywords_en": s.get("keywords_en", [])[:5],
        "domain_tags": s.get("domain_tags", [])[:3],
        "search_tips": s.get("search_tips", ""),
    }


def _tool_export_report(topic_id: str) -> str:
    from topic_manager import active_topics

    state = active_topics.get(topic_id)
    if not state:
        return "课题不存在"
    lines = [
        f"# {state.query}",
        "",
        f"论文数: {state.total_papers}",
        f"状态: {state.status.value}",
    ]
    return "\n".join(lines)


def _tool_get_state(topic_id: str) -> dict:
    from topic_manager import active_topics

    s = active_topics.get(topic_id)
    if not s:
        return {"error": "课题不存在"}
    return {
        "query": s.query,
        "status": s.status.value,
        "papers": s.total_papers,
        "chunks": s.uploaded_papers,
    }


# ---- 工具注册表 ----

TOOLS: list[Tool] = [
    Tool(
        "generate_search_strategy",
        "为用户的研究方向生成检索策略：关键词、检索式、推荐数据库",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "研究方向描述"}},
            "required": ["query"],
        },
        _tool_generate_strategy,
    ),
    Tool(
        "search_papers",
        "多源搜索学术论文（arXiv、OpenAlex、CNKI等），返回论文列表和OA状态",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "keywords_en": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "英文关键词",
                },
                "keywords_cn": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "中文关键词",
                },
            },
            "required": ["query"],
        },
        _tool_search_papers,
    ),
    Tool(
        "download_and_index",
        "下载OA论文并构建LanceDB知识库（最多5篇）",
        {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "PDF下载链接列表",
                },
                "topic_id": {"type": "string", "description": "课题ID"},
            },
            "required": ["urls", "topic_id"],
        },
        _tool_download_index,
        need_confirm=True,
    ),
    Tool(
        "ask_knowledge_base",
        "基于已构建的知识库进行深度学术问答，返回带引用的答案",
        {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "LanceDB表名"},
                "question": {"type": "string", "description": "学术问题"},
            },
            "required": ["table_name", "question"],
        },
        _tool_ask_kb,
    ),
    Tool(
        "export_report",
        "导出当前课题的研究报告为Markdown格式",
        {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string", "description": "课题ID"},
            },
            "required": ["topic_id"],
        },
        _tool_export_report,
    ),
    Tool(
        "get_topic_state",
        "查看当前课题的状态：论文数、知识库就绪状态",
        {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string", "description": "课题ID"},
            },
            "required": ["topic_id"],
        },
        _tool_get_state,
    ),
]


def get_tool(name: str) -> Tool | None:
    for t in TOOLS:
        if t.name == name:
            return t
    return None


def get_tools_schema() -> list[dict]:
    return [t.to_openai_schema() for t in TOOLS]

# ---- 补充工具：搜索已构建的知识库 ----

async def _tool_search_knowledge_bases(query: str) -> dict:
    """搜索已构建的知识库列表，按大小排序，优先返回内容最丰富的。"""
    import lancedb
    from config import LANCEDB_DIR
    db = lancedb.connect(str(LANCEDB_DIR))
    tables = db.table_names()
    kb_tables = [t for t in tables if t.startswith("papers_eval") or t.startswith("eval_")]
    # 按表名长度和内容推断：50papers > llm > 其他
    best = [t for t in kb_tables if "50papers" in t or "eval_llm" in t or "large" in t]
    rest = [t for t in kb_tables if t not in best]
    return {"best_kb": best[0] if best else (kb_tables[0] if kb_tables else ""),
            "all_kbs": kb_tables, "total": len(kb_tables),
            "tip": f"推荐使用 knowledge_base={best[0] if best else 'N/A'}，这是内容最丰富的知识库"}


async def _tool_agent_ask(table_name: str, question: str) -> dict:
    """基于指定知识库问专业学术问题。table_name 从 search_knowledge_bases 获取。"""
    from qa_engine import ask_question
    result = await ask_question(table_name, question)
    refs = [{"title": r.get("title",""), "year": r.get("year")} for r in result.references[:5]]
    return {
        "answer": result.answer[:2000],
        "references": refs,
        "has_supplement": bool(result.supplement),
    }

TOOLS.append(Tool("search_knowledge_bases",
    "搜索本地已构建的知识库列表，返回可用的知识库名称",
    {"type": "object", "properties": {"query": {"type": "string", "description": "任意搜索词"}}, "required": ["query"]},
    _tool_search_knowledge_bases))

TOOLS.append(Tool("agent_ask",
    "基于知识库进行深度学术问答（需要先通过 search_knowledge_bases 获取可用知识库名称）",
    {"type": "object", "properties": {
        "table_name": {"type": "string", "description": "知识库表名"},
        "question": {"type": "string", "description": "学术问题"},
    }, "required": ["table_name", "question"]},
    _tool_agent_ask))
