"""Agent 工具注册表：将现有功能封装为标准 Tool 接口（Function Calling 规范）。

每个工具声明：name, description, parameters(JSON Schema), execute 函数。
"""

import asyncio
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
        """执行工具：类型校验 → 幂等控制 → 自动重试。
        准备2 §工具调用优化：Schema校验 + 参数补全 + 自动重试修复。
        DeepSeek 审查 #1: 退避改用 await asyncio.sleep, 不阻塞事件循环。
        #6: need_confirm 工具执行时记录审计日志 (真实人工闸门需审批 UI)。"""
        # 类型校验：验证必填参数
        required = self.parameters.get("required", [])
        for r in required:
            if r not in kwargs or kwargs[r] is None:
                return f"❌ 缺少必填参数: {r}"

        # 自动重试 (max 2)
        last_error = None
        for attempt in range(3):
            try:
                result = (
                    await self.func(**kwargs)
                    if inspect.iscoroutinefunction(self.func)
                    else self.func(**kwargs)
                )
                result_str = (
                    json.dumps(result, ensure_ascii=False, indent=2)
                    if isinstance(result, (dict, list))
                    else str(result)[:3000]
                )
                if self.need_confirm:
                    logger.warning(
                        f"Confirmed tool executed: {self.name} (args: {list(kwargs)})"
                    )
                return result_str
            except Exception as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))  # 退避
                    continue
        return f"❌ 工具执行失败（重试3次后）: {last_error}"


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
        # DeepSeek 审查 #2: parse_pdf/chunk_text 是 CPU 密集同步操作,
        # 放到线程池执行, 避免阻塞 asyncio 事件循环
        result = await asyncio.to_thread(parse_pdf, content, fname)
        pid = uuid.uuid4().hex[:12]
        papers.append(PaperMeta(paper_id=pid, title=fname, is_oa=True))
        c = await asyncio.to_thread(chunk_text, result["full_text"])
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


def _tool_export_report(topic_id: str) -> dict:
    """导出课题报告。统一返回 dict, 与其他工具格式一致 (DeepSeek 审查 #8)。"""
    from topic_manager import active_topics

    state = active_topics.get(topic_id)
    if not state:
        return {"report": "课题不存在"}
    lines = [
        f"# {state.query}",
        "",
        f"论文数: {state.stats.total_papers}",
        f"状态: {state.status.value}",
    ]
    return {"report": "\n".join(lines)}


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


def _tool_check_draft(draft: str, goal: str = "") -> dict:
    """初稿质量清单自检 (选题→初稿 验收线)。"""
    from draft_quality import check_draft

    res = check_draft(draft, [], goal)
    res["summary"] = (
        f"通过 {res['passed']}/{res['total']} 项，判定: {res['verdict']}。"
        " 有未达标项必须修订后再交付。"
    )
    return res


async def _tool_analyze_gap(topic: str) -> dict:
    """研究空白分析 (选题可行性): 真实检索 + 综合, 回答「这个选题有没有人做过」。
    async (修复: 原 sync 内 asyncio.run() 在 agent 异步上下文调用会崩 running event loop)。"""
    from research_gap import analyze_gap

    try:
        return await analyze_gap(topic)
    except Exception as e:
        return {"topic": topic, "error": str(e)}


def _tool_review_draft(draft: str, goal: str = "") -> dict:
    """初稿审稿与批注: 模拟同行评审 5 维评分 + 逐条批注。"""
    from draft_review import review_draft

    return review_draft(draft, goal)


async def _tool_citation_chain(paper_id: str, direction: str = "both") -> dict:
    """引用链追踪: 查论文的引用/被引关系, 理清研究脉络。"""
    from integrations import expand_citation_chain

    try:
        return await expand_citation_chain(paper_id, direction=direction, depth=1)
    except Exception as e:
        return {"error": str(e)}


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
    Tool(
        "check_draft_quality",
        "初稿质量清单自检: 论文初稿是否达到可交付标准 (研究问题/空白定位/结构/引用/诚实性/可续写共6项)。"
        "写论文初稿完成前必须调用; 有未达标项则修订后再次调用, 直至全部通过。",
        {
            "type": "object",
            "properties": {
                "draft": {"type": "string", "description": "论文初稿全文 (Markdown)"},
                "goal": {"type": "string", "description": "研究目标/选题 (可选)"},
            },
            "required": ["draft"],
        },
        _tool_check_draft,
    ),
    Tool(
        "analyze_research_gap",
        "研究空白分析: 输入选题, 真实检索后给出研究空白与可行性报告, 回答「这个选题有没有人做过」",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "选题/研究方向"}
            },
            "required": ["topic"],
        },
        _tool_analyze_gap,
    ),
    Tool(
        "review_draft",
        "初稿审稿与批注: 模拟同行评审, 5 维评分(技术正确性/实验说服力/贡献清晰度/写作质量/可复现性) + 逐条批注 + 待澄清问题",
        {
            "type": "object",
            "properties": {
                "draft": {"type": "string", "description": "论文初稿全文 (Markdown)"},
                "goal": {"type": "string", "description": "研究目标/选题 (可选)"},
            },
            "required": ["draft"],
        },
        _tool_review_draft,
    ),
    Tool(
        "citation_chain",
        "引用链追踪: 输入论文ID, 返回其引用/被引关系网络, 理清研究脉络与关键节点",
        {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "论文 ID"},
                "direction": {
                    "type": "string",
                    "description": "citations=查引用了哪些, references=查被谁引用, both=两者",
                    "enum": ["both", "citations", "references"],
                },
            },
            "required": ["paper_id"],
        },
        _tool_citation_chain,
    ),
]


def register(tool: Tool) -> bool:
    """动态注册工具 (插件机制): 同名覆盖, 幂等。"""
    for i, t in enumerate(TOOLS):
        if t.name == tool.name:
            TOOLS[i] = tool
            return True
    TOOLS.append(tool)
    return True


def _load_plugin_tools() -> int:
    """从 agent/tools_extra/*.py 动态加载工具 (每个文件导出 TOOLS 列表)。
    新增工具只需丢一个文件到 tools_extra/, 无需改注册表代码。
    这就是"智能动态加载和注册": 外部工具插件自动发现。"""
    import importlib.util
    from pathlib import Path

    extra_dir = Path(__file__).resolve().parent / "tools_extra"
    if not extra_dir.exists():
        return 0
    count = 0
    for py in sorted(extra_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"tools_extra.{py.stem}", py
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            tools = getattr(mod, "TOOLS", []) or []
            for t in tools:
                register(t)
                count += 1
            if tools:
                logger.info(f"Plugin tools loaded from {py.name}: {len(tools)}")
        except Exception as e:
            logger.warning(f"Plugin tool load failed: {py.name}: {e}")
    return count


_load_plugin_tools()


def get_tool(name: str) -> Tool | None:
    for t in TOOLS:
        if t.name == name:
            return t
    return None


def get_tools_schema() -> list[dict]:
    return [t.to_openai_schema() for t in TOOLS]


# ---- 补充工具：搜索已构建的知识库 ----


async def _tool_search_knowledge_bases(query: str = "") -> dict:
    """搜索已构建的知识库列表，按大小排序，优先返回内容最丰富的。
    query 为占位参数 (DeepSeek 审查 #5/#10: 当前不参与过滤, 保持可选)。"""
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    tables = db.table_names()
    # 只列出用户构建的知识库（papers_* 前缀），排除评估表
    kb_tables = [
        t for t in tables if (t.startswith("papers_") and "eval" not in t.lower())
    ]
    if not kb_tables:
        return {"best_kb": "", "all_kbs": [], "total": 0, "tip": "暂无知识库，请先上传 PDF 构建知识库"}
    # 按行数排序
    scored = []
    for t in kb_tables:
        try:
            n = db.open_table(t).count_rows()
            scored.append((n, t))
        except Exception:
            scored.append((0, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {
        "best_kb": scored[0][1] if scored else "",
        "all_kbs": [t for _, t in scored],
        "total": len(scored),
        "tip": f"推荐使用 knowledge_base={scored[0][1]}，共 {scored[0][0]} 行",
    }


async def _tool_agent_ask(table_name: str, question: str) -> dict:
    """基于指定知识库问专业学术问题。table_name 从 search_knowledge_bases 获取。"""
    from qa_engine import ask_question

    result = await ask_question(table_name, question)
    refs = [
        {"title": r.get("title", ""), "year": r.get("year")}
        for r in result.references[:5]
    ]
    return {
        "answer": result.answer[:2000],
        "references": refs,
        "has_supplement": bool(result.supplement),
    }


# DeepSeek 审查 #4: 幂等注册, 防止模块重复导入/热重载时工具累积重复项
if not any(t.name == "search_knowledge_bases" for t in TOOLS):
    TOOLS.append(
        Tool(
            "search_knowledge_bases",
            "搜索本地已构建的知识库列表，返回可用的知识库名称（query 为占位参数）",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "占位参数（可选，暂不参与过滤）",
                    }
                },
            },
            _tool_search_knowledge_bases,
        )
    )

if not any(t.name == "agent_ask" for t in TOOLS):
    TOOLS.append(
        Tool(
            "agent_ask",
            "基于知识库进行深度学术问答（需要先通过 search_knowledge_bases 获取可用知识库名称）",
            {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "知识库表名"},
                    "question": {"type": "string", "description": "学术问题"},
                },
                "required": ["table_name", "question"],
            },
            _tool_agent_ask,
        )
    )
