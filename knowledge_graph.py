"""论文知识图谱：实体抽取 + 关系构建 + JSON 导出 + D3.js 可视化。

准备2(技术) §54: 用知识图谱为精准数据兜底，搭配向量数据库。
准备2(技术) §1507-1512: GraphRAG 以向量检索做语义初筛，叠加图拓扑推理。
准备2(技术) §1424: 向量检索甚至纯文件系统就够，无需一开始就用重型方案
→ 本模块选用轻量 JSON 图 + 前端 D3.js 可视化，不引入 Neo4j。"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ===== 实体抽取 Prompt =====
EXTRACT_PROMPT = """从以下论文摘要中抽取关键实体和关系。输出 JSON，不要解释。

{abstracts}

JSON 格式：
{{
  "entities": [
    {{"id": "方法名/数据集/指标", "type": "method|dataset|metric", "label": "显示名"}}
  ],
  "relations": [
    {{"source": "entity_id", "target": "entity_id", "relation": "evaluated_on|outperforms|based_on|uses|compared_with"}}
  ]
}}

规则：
- method: 算法、模型、框架名称
- dataset: 数据集、基准名称
- metric: 评估指标
- 只抽取摘要中明确提到的实体"""


async def extract_entities_from_abstracts(abstracts: list[dict[str, str]]) -> dict:
    """用 DeepSeek 从论文摘要批量抽取实体关系。"""
    from openai import AsyncOpenAI

    from config import (
        DEEPSEEK_API_KEY,
        DEEPSEEK_BASE_URL,
        DEEPSEEK_MODEL,
        LLM_STRATEGY_MAX_TOKENS,
        LLM_STRATEGY_TEMPERATURE,
    )
    from monitor import safe_call

    if not abstracts:
        return {"entities": [], "relations": []}

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    text = "\n\n".join(
        f"[{a.get('paper_id','?')}] {a.get('title','')}: {a.get('abstract','')[:500]}"
        for a in abstracts[:20]
    )
    try:
        resp = await safe_call(
            client.chat.completions.create,
            model=DEEPSEEK_MODEL,
            temperature=LLM_STRATEGY_TEMPERATURE,
            max_tokens=LLM_STRATEGY_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(abstracts=text[:8000]),
                }
            ],
            max_retries=2,
            timeout=30.0,
            extra_body={"thinking": {"type": "disabled"}},  # 推理模型禁 thinking, 否则 JSON 输出被 reasoning 挤空
            source="llm",
        )
        if resp is None:
            return {"entities": [], "relations": []}
        content = resp.choices[0].message.content.strip()
        # Clean markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return {"entities": [], "relations": []}


def build_graph_from_kb(table_name: str) -> dict:
    """从 LanceDB 知识库构建图谱（同步，不含 LLM 抽取）。
    返回 D3.js 兼容的 force-graph JSON。

    DeepSeek 审查:
      #1: 补充共同作者关系边, 使图谱成为真实的合作网络 (原实现 edges 恒空,
          只有孤立论文节点, 不成其为知识图谱)。
      #2: 简化 LanceDB 表列举兼容逻辑 (去掉脆弱的 tuple 解构)。
      #6: 用 is_fulltext=false 过滤扫描取"每篇论文一条摘要", 替代无查询 search()。
    """
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        raw = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
        tables = list(raw.tables) if hasattr(raw, "tables") else list(raw)
    except Exception:
        return _empty_graph()

    if table_name not in [str(t) for t in tables]:
        return _empty_graph()

    try:
        tbl = db.open_table(table_name)
        # 每篇论文恰好一条摘要记录 (is_fulltext=false) → 确定性过滤扫描, 无需去重
        rows = tbl.search().where("is_fulltext = false").limit(200).to_list()
    except Exception as e:
        logger.warning(f"KG build failed for {table_name}: {e}", exc_info=True)
        return _empty_graph()

    # 构建节点 + 共同作者关系边
    nodes, links = [], []
    author_to_papers: dict[str, list[str]] = {}
    for r in rows:
        pid = r.get("paper_id", "")
        if not pid:
            continue
        title = (r.get("title") or pid)[:80]
        year = r.get("year", "")
        nodes.append(
            {
                "id": pid,
                "label": title,
                "group": "paper",
                "year": str(year) if year else "",
            }
        )
        # 作者索引 (逗号/分号/and 分隔)
        for a in re.split(r"[,;]|\band\b", (r.get("authors") or "")):
            a = a.strip().strip(".")
            if len(a) < 2 or not any(c.isalpha() for c in a):
                continue
            author_to_papers.setdefault(a, []).append(pid)

    # 共同作者边 (上限 300, 防 D3 force-graph 卡顿)
    seen_edges: set[tuple[str, str]] = set()
    for author, pids in author_to_papers.items():
        if len(pids) < 2:
            continue
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                key = tuple(sorted([pids[i], pids[j]]))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                links.append(
                    {
                        "source": pids[i],
                        "target": pids[j],
                        "relation": "shared_author",
                        "label": author,
                    }
                )
                if len(links) >= 300:
                    break
            if len(links) >= 300:
                break
        if len(links) >= 300:
            break

    return {"nodes": nodes, "links": links}


def _empty_graph():
    return {"nodes": [], "links": []}


