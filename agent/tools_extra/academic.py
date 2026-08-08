"""学术工具插件 — 动态加载示例。

放在 agent/tools_extra/ 下, 注册表启动时自动发现注册。
新增工具 = 在此文件导出 TOOLS 列表 (或新建 .py 文件导出 TOOLS), 无需改注册表。

用法示例: python -c "from agent.tool_registry import get_tool; print(get_tool('doi_lookup'))"
"""
from agent.tool_registry import Tool


async def _doi_lookup(doi: str) -> dict:
    """DOI → 论文元数据 (Crossref API)。"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"https://api.crossref.org/works/{doi}")
            if r.status_code == 200:
                m = r.json().get("message", {})
                authors = ", ".join(
                    f"{a.get('given', '')} {a.get('family', '')}"
                    for a in (m.get("author") or [])[:5]
                )
                return {
                    "title": (m.get("title") or [""])[0],
                    "authors": authors,
                    "year": (m.get("issued", {}).get("date-parts", [[None]])[0][0]),
                    "doi": doi,
                    "container": (m.get("container-title") or [""])[0],
                }
        return {"error": f"DOI {doi} 未找到"}
    except Exception as e:
        return {"error": str(e)}


async def _arxiv_lookup(arxiv_id: str) -> dict:
    """arXiv ID → 论文元数据 (标题/摘要)。"""
    import xml.etree.ElementTree as ET

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://export.arxiv.org/api/query", params={"id_list": arxiv_id}
            )
            root = ET.fromstring(r.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entry = root.find("a:entry", ns)
            if entry is None:
                return {"error": f"arXiv {arxiv_id} 未找到"}
            title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": (entry.findtext("a:summary", "", ns) or "").strip()[:500],
            }
    except Exception as e:
        return {"error": str(e)}


TOOLS = [
    Tool(
        "doi_lookup",
        "DOI 查论文: 输入 DOI, 返回标题/作者/年份/期刊 (Crossref)",
        {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI, 如 10.1109/tpami.2018.2798607"}
            },
            "required": ["doi"],
        },
        _doi_lookup,
    ),
    Tool(
        "arxiv_lookup",
        "arXiv 查论文: 输入 arXiv ID, 返回标题/摘要",
        {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID, 如 1706.03762"}
            },
            "required": ["arxiv_id"],
        },
        _arxiv_lookup,
    ),
]
