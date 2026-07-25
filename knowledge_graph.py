"""论文知识图谱：实体抽取 + 关系构建 + JSON 导出 + D3.js 可视化。

准备2(技术) §54: 用知识图谱为精准数据兜底，搭配向量数据库。
准备2(技术) §1507-1512: GraphRAG 以向量检索做语义初筛，叠加图拓扑推理。
准备2(技术) §1424: 向量检索甚至纯文件系统就够，无需一开始就用重型方案
→ 本模块选用轻量 JSON 图 + 前端 D3.js 可视化，不引入 Neo4j。"""

import json
import logging
from pathlib import Path

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

    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
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
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(abstracts=text[:8000]),
                }
            ],
            max_retries=1,
            timeout=30.0,
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
    返回 D3.js 兼容的 force-graph JSON。"""
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        raw = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
        tables = (
            raw.tables
            if hasattr(raw, "tables")
            else (raw[0] if isinstance(raw, tuple) else raw)
        )
    except Exception:
        return _empty_graph()

    if table_name not in [str(t) for t in tables]:
        return _empty_graph()

    try:
        tbl = db.open_table(table_name)
        # 查前 200 行构建节点
        rows = tbl.search().limit(200).to_list()
        # 按 paper_id 去重
        seen = set()
        deduped = []
        for r in rows:
            pid = r.get("paper_id", "")
            if pid and pid not in seen:
                seen.add(pid)
                deduped.append(r)
        rows = deduped
    except Exception as e:
        logger.warning(f"KG build failed for {table_name}: {e}")
        return _empty_graph()

    # 构建节点
    nodes, edges = [], []
    for r in rows:
        pid = r.get("paper_id", "")
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

    return {"nodes": nodes, "links": edges}


def _empty_graph():
    return {"nodes": [], "links": []}


def save_graph_json(graph: dict, path: str = "data/knowledge_graph.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    logger.info(
        f"Graph saved to {path} ({len(graph['nodes'])} nodes, {len(graph['links'])} edges)"
    )


def load_graph_json(path: str = "data/knowledge_graph.json") -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return _empty_graph()
