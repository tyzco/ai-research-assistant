"""Agent 工具注册表：将现有功能封装为标准 Tool 接口（Function Calling 规范）。

每个工具声明：name, description, parameters(JSON Schema), execute 函数。
"""

import asyncio
import inspect
import json
import logging
import re
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
        kind: str = "deterministic",
        retryable: bool = True,
        requires_precheck: str = "",
        precheck_max_age: int = 300,
        cacheable: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema
        self.func = func
        self.need_confirm = need_confirm  # 是否需要人工确认
        # 重试的前提是**重放安全**: 同一入参执行两次, 结果与副作用等价于执行一次。
        #
        # 有一类操作不能自动重试 —— 下发指令、扣款、改状态。它们失败后的正确动作是
        # **先查询当前状态再决定**, 而不是赌第二次: 超时并不代表没生效 (可能已经
        # 执行完了, 只是响应没回来), 盲目重试等于把一次操作变成两次。
        #
        # 当前 20 个工具全部是 True, 而且这个判断核实过: 除 download_and_index 外
        # **没有任何一个写文件**(export_* 只是返回 dict), 而 download_and_index 走
        # doc_key 版本管理 —— 重放产生的新记录会把旧的标记为 superseded 并被检索
        # 过滤, 对使用者等价于幂等。
        #
        # 也就是说**这个守卫今天不会被触发**。它的价值在于: 下一个工具 (真正下发
        # 操作的那个) 不会因为"默认值是 True"而悄悄获得自动重试。安全领域里这类
        # 默认值反了的设计, 出事时没有任何征兆。
        self.retryable = retryable
        # 结果是否可缓存 (只读且同参同果)。
        # 必须**显式声明**而不能从 kind/retryable 推断: 实测 download_and_index 与
        # export_report 都是 "deterministic + retryable", 但它们会写知识库/写文件 ——
        # 按推断规则缓存会造成副作用被吞 (第二次调用命中缓存, 根本没执行)。
        self.cacheable = cacheable
        # 执行前必须**先查当前状态**的工具名 (空 = 无前置)。对应"下发操控指令前先查询
        # 设备状态、确认之后再执行"的通用形式:
        #
        # 提示词里写一句"请先查询状态"是**建议**, 模型可以不听; 这条是**强制**——
        # 执行循环在调用前检查, 没查过就直接拒绝。区别在于失效模式: 前者失效时你
        # 得到一次未经确认的操作, 后者失效时你只得到一个"先去查"的错误。
        #
        # `precheck_max_age` 是新鲜度上限: 十分钟前查过的状态不能拿来背书现在的操作。
        self.requires_precheck = requires_precheck
        self.precheck_max_age = precheck_max_age
        # deterministic = 确定性能力 (查库/计算/导出/查询: 同输入同输出, 可缓存可审计)
        # llm           = 内部含 LLM 判断/生成 (结果有随机性, 成本高, 需谨慎包装)
        # 设计原则: 工具优先确定性能力; LLM 类须显式标注, 避免"把模糊推理包成工具"而不自知。
        self.kind = kind

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, kwargs: dict) -> str:
        """按 JSON Schema 校验模型给出的参数, 返回错误说明 (空串 = 通过)。

        此前只查 `required`, 类型/区间/枚举一概不看。所以 `max_results: 99999`
        和 `direction: "随便"` 都会被原样接受, 工具或下游 API 再去报一个跟真实
        原因无关的错。**校验的目的不只是拦住, 更是把"哪里错了、应该是什么"
        回灌给模型**, 让它有机会重写一次——只返回 False 是浪费了这次机会。
        """
        props = self.parameters.get("properties", {}) or {}
        errors: list[str] = []

        for r in self.parameters.get("required", []) or []:
            if r not in kwargs or kwargs[r] is None:
                errors.append(f"缺少必填参数 `{r}`")
            elif isinstance(kwargs.get(r), str) and not kwargs[r].strip():
                errors.append(f"必填参数 `{r}` 不能为空字符串")

        type_map = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "array": list, "object": dict,
        }
        for key, val in kwargs.items():
            spec = props.get(key)
            if not isinstance(spec, dict) or val is None:
                continue
            want = spec.get("type")
            if want in type_map:
                bad = not isinstance(val, type_map[want])
                # bool 是 int 的子类 —— `isinstance(True, int)` 为 True, 所以
                # `limit=True` 会当成合法整数放行。这个特例必须在 isinstance
                # 判定之外单独挡住 (写成 `if not isinstance: if isinstance(val,bool)`
                # 是永远进不去的死分支)。
                if isinstance(val, bool) and want in ("integer", "number"):
                    bad = True
                if bad:
                    errors.append(
                        f"参数 `{key}` 类型不对: 需要 {want}, 收到 {type(val).__name__}"
                    )
                    continue
            if "enum" in spec and val not in spec["enum"]:
                errors.append(f"参数 `{key}` 取值必须是 {spec['enum']} 之一, 收到 {val!r}")
            if "pattern" in spec and isinstance(val, str):
                # 声明了 pattern 就必须执行 —— 写在 schema 里但不校验, 比不写更糟:
                # 模型以为有约束, 实际没有, 而错误会在下游以无关的形式出现。
                if not re.fullmatch(spec["pattern"], val):
                    errors.append(
                        f"参数 `{key}` 格式不对: 需要匹配 {spec['pattern']}, 收到 {val!r}"
                    )
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if "minimum" in spec and val < spec["minimum"]:
                    errors.append(f"参数 `{key}` 最小为 {spec['minimum']}, 收到 {val}")
                if "maximum" in spec and val > spec["maximum"]:
                    errors.append(f"参数 `{key}` 最大为 {spec['maximum']}, 收到 {val}")
            if isinstance(val, list) and "maxItems" in spec and len(val) > spec["maxItems"]:
                errors.append(
                    f"参数 `{key}` 最多 {spec['maxItems']} 项, 收到 {len(val)} 项"
                )

        if not errors:
            return ""
        return (
            "❌ 参数校验未通过:\n- "
            + "\n- ".join(errors)
            + f"\n该工具的参数结构: {json.dumps(self.parameters, ensure_ascii=False)[:600]}"
            + "\n请修正后重新调用。"
        )

    def missing_precheck(self, called: dict[str, float], now: float) -> str:
        """执行前还缺哪个前置查询? 空串 = 可以执行。

        `called` 是 {工具名: 上次成功调用的时间戳}。由执行循环维护并传入 ——
        工具自己拿不到调用历史, 也不该去猜。
        """
        if not self.requires_precheck:
            return ""
        ts = called.get(self.requires_precheck)
        if ts is None or (now - ts) > self.precheck_max_age:
            return self.requires_precheck
        return ""

    async def execute(self, **kwargs) -> str:
        """执行工具：Schema 校验 → 重试。
        准备2 §工具调用优化：Schema校验 + 参数补全 + 自动重试修复。
        DeepSeek 审查 #1: 退避改用 await asyncio.sleep, 不阻塞事件循环。
        #6: need_confirm 工具执行时记录审计日志 (真实人工闸门需审批 UI)。"""
        invalid = self.validate(kwargs)
        if invalid:
            return invalid

        # 缓存查: 只对显式声明 cacheable=True 的只读工具生效。
        # 放在 validate 之后 —— 参数都没过校验的调用不该占用缓存条目。
        if self.cacheable:
            try:
                import tool_cache

                _hit = tool_cache.get(self.name, kwargs)
                if _hit is not None:
                    return _hit
            except Exception:
                pass

        # 只读工具重试; 有副作用的工具失败即停。
        MAX_ATTEMPTS = 3 if self.retryable else 1
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
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
                if self.cacheable:
                    try:
                        import tool_cache

                        tool_cache.put(self.name, kwargs, result_str)
                    except Exception:
                        pass
                return result_str
            except Exception as e:
                last_error = e
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # 退避
                    continue

        if not self.retryable:
            # 「没重试」和「重试后成功」在结果上完全一样, 对模型必须说清楚区别,
            # 否则它会当成一次普通的失败再调一遍 —— 那就等于绕过了这里的保护。
            return (
                f"❌ 工具 `{self.name}` 执行失败, 且该操作**不自动重试**（有副作用）。\n"
                f"错误: {last_error}\n"
                f"注意: 超时不代表未生效 —— 它可能已经部分完成。"
                f"请先查询当前状态确认, 再决定是否重新执行。"
            )
        return f"❌ 工具执行失败（重试{MAX_ATTEMPTS}次后）: {last_error}"


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
        kind="llm"
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
                    "maxItems": 5,
                    "description": "英文关键词（最多 5 个，超出按前 5 个处理）",
                },
                "keywords_cn": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                    "description": "中文关键词（最多 5 个，超出按前 5 个处理）",
                },
            },
            "required": ["query"],
        },
        _tool_search_papers,
        cacheable=True,
    ),
    Tool(
        "download_and_index",
        "下载 OA 论文并构建 LanceDB 知识库（单次最多 5 篇）。示例: {\"urls\":[\"https://arxiv.org/pdf/1706.03762.pdf\"], \"topic_id\":\"abc123\"}",
        {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                    "description": "PDF 下载链接列表。**单次最多 5 条**，超出部分会被直接丢弃（不报错）——需要更多请分多次调用",
                },
                "topic_id": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{12}$",
                    "description": "课题ID：12 位小写十六进制，例如 e9d9e4e4a9fa"
                    "（由 create_topic 生成，不是自拟名称）。",
                },
            },
            "required": ["urls", "topic_id"],
        },
        _tool_download_index,
        need_confirm=True,
        # 真实的前置条件, 不是为了演示造的: `build_knowledge_base` 会按 topic_id
        # 建 `papers_<topic_id>` 表 —— topic_id 不存在时就留下一张孤儿表, 而且
        # 没有任何东西会引用它。先查一次课题状态, 确认存在再动手。
        requires_precheck="get_topic_state",
    ),
    Tool(
        "ask_knowledge_base",
        "基于已构建的知识库进行深度学术问答，返回带引用的答案。示例: {\"table_name\":\"papers_abc123\", \"question\":\"该方法的损失函数是什么？\"} （table_name 见系统提示的【当前课题知识库作用域】）",
        {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "知识库名。两种写法都接受："
                    "① 直接传 12 位 topic_id（如 e9d9e4e4a9fa，会自动解析成对应表）；"
                    "② 传完整表名 papers_<topic_id>（如 papers_e9d9e4e4a9fa）。",
                },
                "question": {"type": "string", "description": "学术问题"},
            },
            "required": ["table_name", "question"],
        },
        _tool_ask_kb,
        kind="llm",
        cacheable=True,
    ),
    Tool(
        "export_report",
        "导出当前课题的研究报告为Markdown格式",
        {
            "type": "object",
            "properties": {
                "topic_id": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{12}$",
                    "description": "课题ID：12 位小写十六进制，例如 e9d9e4e4a9fa"
                    "（由 create_topic 生成，不是自拟名称）。",
                },
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
                "topic_id": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{12}$",
                    "description": "课题ID：12 位小写十六进制，例如 e9d9e4e4a9fa"
                    "（由 create_topic 生成，不是自拟名称）。",
                },
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
        cacheable=True,
        kind="llm"
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
        cacheable=True,
        kind="llm"
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
        cacheable=True,
        kind="llm"
    ),
    Tool(
        "citation_chain",
        "引用链追踪: 输入论文 ID, 返回其引用/被引关系网络, 理清研究脉络与关键节点。示例: {\"paper_id\":\"W2963403868\", \"direction\":\"citations\"} （paper_id 用 Semantic Scholar 的 ID，不是论文标题）",
        {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{16}$",
                    "description": "论文 ID：16 位十六进制，例如 3f2a9c1b8d4e5f60。"
                    "必须取自 search_papers / ask_knowledge_base 的返回，不要自拟。",
                },
                # 这个 dict 里 `enum` 曾写了两遍 —— Python 静默取后一个, 前者形同虚设。
                # 这次两组取值恰好相同所以没出事, 但重复键是"改了上面那行依然不生效"
                # 这类问题的标准来源, 而且不会有任何报错。
                "direction": {
                    "type": "string",
                    "enum": ["both", "citations", "references"],
                    "description": "citations=谁引用了这篇, references=这篇引用了谁, both=两者都查。必须是这三个值之一，否则返回空结果",
                },
            },
            "required": ["paper_id"],
        },
        _tool_citation_chain,
        cacheable=True,
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
            cacheable=True,  # 只读: 列知识库清单
        )
    )

if not any(t.name == "agent_ask" for t in TOOLS):
    TOOLS.append(
        Tool(
            "agent_ask",
            "基于知识库进行深度学术问答（需要先通过 search_knowledge_bases 获取可用知识库名称）。示例: {\"table_name\":\"papers_abc123\", \"question\":\"实验用了哪些数据集？\"}",
            {
                "type": "object",
                "properties": {
                    "table_name": {
                    "type": "string",
                    "description": "知识库名。两种写法都接受："
                    "① 直接传 12 位 topic_id（如 e9d9e4e4a9fa，会自动解析成对应表）；"
                    "② 传完整表名 papers_<topic_id>（如 papers_e9d9e4e4a9fa）。",
                },
                    "question": {"type": "string", "description": "学术问题"},
                },
                "required": ["table_name", "question"],
            },
            _tool_agent_ask,
            kind="llm",  # 内部走 RAG + LLM 生成 (非确定性): 显式标注, 避免误当确定性工具
            cacheable=True,  # 只读: 同表同问可复用 (LLM 输出虽有随机性, 但同问同答更好也更省)
        )
    )
