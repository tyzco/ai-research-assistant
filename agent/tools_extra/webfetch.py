"""联网工具插件 — 读链接提取信息 (fetch_url)。

放在 agent/tools_extra/ 下, 注册表启动时自动发现注册。
依赖: httpx + bs4 + pymupdf (均已在环境)。

能力:
  1. 读网页: httpx 抓取 → bs4 提正文 (去导航/脚本/广告/页脚)
  2. 读 PDF 链接: 下载 → parse_pdf 提全文 (复用 downloader)
用途: 用户贴链接, agent 读取信息/判断价值, 决定是否 download_and_index 建库。

边界 (合法红线): 只读用户提供的链接 + OA 公开学术页面, 不抓付费/盗版全文。
"""
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_MAX_TEXT = 4000  # 返回正文上限, 防止撑爆 agent 上下文


async def _extract_pdf_text(content: bytes) -> dict:
    """PDF 链接 → 全文提取 (parse_pdf 是 CPU 密集同步函数 → 线程池)。"""
    from downloader import parse_pdf

    result = await asyncio.to_thread(parse_pdf, content, "fetch_url.pdf")
    full = (result.get("full_text") or "").strip()
    return {
        "content_type": "pdf",
        "content": full[:_MAX_TEXT] or "(PDF 无可提取文本)",
        "truncated": len(full) > _MAX_TEXT,
    }


def _extract_html(url: str, html_text: str) -> dict:
    """网页 → 去噪提取正文 (bs4, 去脚本/导航/页脚等非正文节点)。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "form", "svg", "button", "iframe"]
    ):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n", strip=True).split("\n") if ln.strip()]
    body = "\n".join(lines)
    return {
        "title": title[:300],
        "content_type": "html",
        "content": body[:_MAX_TEXT] or "(页面无可提取正文)",
        "truncated": len(body) > _MAX_TEXT,
        "url": url[:300],
    }


async def _fetch_url(url: str) -> dict:
    """抓取并提取链接内容: 网页返回正文, PDF 返回全文。"""
    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: 无法访问 {url}"}
        ctype = (r.headers.get("content-type") or "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            info = await _extract_pdf_text(r.content)
        else:
            info = _extract_html(url, r.text)
        info["url"] = url[:300]
        return info
    except Exception as e:
        return {"error": f"抓取失败 {url}: {e}"}


from agent.tool_registry import Tool  # noqa: E402  (必须放文件尾, 避免循环导入)

TOOLS = [
    Tool(
        "fetch_url",
        "读取链接内容: 输入 URL, 网页返回标题+正文, PDF 返回全文。"
        "用户贴链接时调用, 判断信息后决定是否 download_and_index 建库。",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要读取的链接, 如 arXiv 摘要页/项目主页/PDF 直链"}
            },
            "required": ["url"],
        },
        _fetch_url,
    ),
]
