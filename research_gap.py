"""研究空白分析（选题可行性）—— 回答「这个选题有没有人做过 / 可行性如何」。

流程: 输入选题 → 真实检索(12 源) → LLM 综合 → 研究空白报告:
  - 已有工作概览（基于检索到的真实文献, 分组归纳）
  - 已覆盖的子问题
  - 研究空白（检索未覆盖、值得做的方向, 锚定到文献）
  - 可行性判断（数据可得性 / 方法成熟度 / 风险）
  - 诚实的不确定性声明（需扩大检索确认, 绝不编造空白）

与初稿质量清单(draft_quality)配合: 空白分析在选题阶段, 质量清单在交付阶段。

自测: python research_gap.py "选题"
"""

import asyncio
import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

_SYNTH_PROMPT_MD = """你是学术研究选题顾问。基于以下【检索到的真实文献】，对选题「{topic}」做研究空白与可行性分析。
直接输出 Markdown 报告（不要 JSON、不要额外解释），必须包含以下章节（用 ## 二级标题）：
- ## 已有工作概览：1-2 句，基于真实文献
- ## 已覆盖的子问题：列表，已充分研究的方向
- ## 研究空白（候选方向）：每个空白写「方向 + 依据哪些文献/未覆盖哪方面 + 置信度(高/中/低)」
- ## 可行性：数据可得性 / 方法成熟度 / 主要风险
- ## 不确定性声明：检索局限（如"基于检索到的 N 篇，需扩大检索/人工复核确认"）

要求：每个空白判断必须基于检索到的文献，并标注不确定性；宁可说"未覆盖/不确定"，也不编造空白。
引用文献时用 [编号] 标注，并在首次提及时附上该文献的链接（Markdown 格式，方便点击直达原文）。

【检索到的文献】
{papers}

【选题】{topic}
"""


def _papers_txt(papers: list) -> str:
    lines = []
    for i, p in enumerate(papers[:50], 1):
        link = ""
        if p.get("doi"):
            link = f"https://doi.org/{p['doi']}"
        elif p.get("arxiv_id"):
            link = f"https://arxiv.org/abs/{p['arxiv_id']}"
        elif p.get("pdf_url"):
            link = p["pdf_url"]
        lines.append(
            f"[{i}] {p.get('title','')} ({p.get('year','')}) · {p.get('source','')}"
            + (f" · {link}" if link else "")
            + (
                f"\n    abstract: {(p.get('abstract') or '')[:200]}"
                if p.get("abstract")
                else ""
            )
        )
    return "\n".join(lines) or "（检索无结果）"


def _references_md(papers: list) -> str:
    """参考文献清单: 每条带可点击 Markdown 链接。OA 优先直链 PDF, 否则 DOI/arXiv。"""
    lines = ["\n## 参考文献（点击直达原文）"]
    for i, p in enumerate(papers[:50], 1):
        title = (p.get("title") or "无标题")[:80]
        link = ""
        if p.get("pdf_url"):
            link = p["pdf_url"]
        elif p.get("doi"):
            link = f"https://doi.org/{p['doi']}"
        elif p.get("arxiv_id"):
            link = f"https://arxiv.org/abs/{p['arxiv_id']}"
        if link:
            lines.append(f"[{i}] [{title}]({link}) ({p.get('year','')})")
        else:
            lines.append(f"[{i}] {title} ({p.get('year','')}) · 闭源，需学校账号下载")
    return "\n".join(lines)


async def _fast_search(topic: str):
    """快速检索: 跳过慢的 LLM 策略生成 (generate_search_strategy), 用缓存翻译。

    降低首字延迟: generate_search_strategy 生成复杂 JSON (关键词/领域/提示) 慢,
    这里只用缓存翻译 (中文→英文关键词) 直接喂给 search_papers_for_topic。
    """
    from search import _translate_cn_query, search_papers_for_topic

    keywords_en = None
    if any(19968 <= ord(c) <= 40869 for c in topic):  # 含中文 → 缓存翻译
        try:
            en_txt = await _translate_cn_query(topic)
            if en_txt:
                keywords_en = [k.strip() for k in en_txt.split(",") if k.strip()]
        except Exception:
            keywords_en = None
    return await search_papers_for_topic(topic, keywords_en=keywords_en)


async def _synthesize(topic: str, papers: list, stream: bool = False):
    """生成 markdown 报告 (用 llm_client flash 模型 + 高上限避免截断)。

    stream=True → 返回 async generator 逐 token (供 SSE 逐字推送);
    stream=False → 返回完整 markdown 字符串。
    """
    from config import DEEPSEEK_MODEL
    from llm_client import get_client

    prompt = _SYNTH_PROMPT_MD.format(topic=topic, papers=_papers_txt(papers))
    client = get_client()

    async def _gen():
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.3,
            max_tokens=4096,  # 高上限, 避免报告截断
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            extra_body={"thinking": {"type": "disabled"}},
        )
        async for chunk in resp:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    if stream:
        return _gen()

    full = []
    async for token in _gen():
        full.append(token)
    return "".join(full).strip()


async def analyze_gap(topic: str, top_k: int = 50, progress_cb=None) -> dict:
    """选题 → 研究空白报告 (markdown)。返回 {topic, n_papers, papers, report, markdown}。

    progress_cb: 可选 async 回调, 在关键里程碑调用 (供 SSE 流式进度推送)。
    检索失败时优雅降级: 返回空报告并标注, 不中断。
    """

    async def _notify(msg: str) -> None:
        if progress_cb:
            try:
                await progress_cb(msg)
            except Exception:
                pass

    await _notify(f"正在检索「{topic}」相关文献…")
    papers = []
    try:
        papers = await _fast_search(topic)
    except Exception as e:
        logger.warning(f"Search failed for gap analysis: {e}")

    await _notify(f"检索到 {len(papers)} 篇，正在综合分析研究空白…")
    meta = [p.model_dump() for p in papers[:top_k]] if papers else []
    try:
        markdown = await _synthesize(topic, meta, stream=False)
    except Exception as e:
        logger.exception(f"Gap synthesis failed: {e}")
        markdown = f"## 检索结果\n检索到 {len(papers)} 篇文献。LLM 综合失败，请稍后重试或人工判断。"
    markdown += _references_md(meta)
    return {
        "topic": topic,
        "n_papers": len(papers),
        "papers": meta,
        "report": markdown,
        "markdown": markdown,
    }


async def stream_analyze(topic: str, top_k: int = 50):
    """流式选题分析: async generator, yield (kind, payload)。

    kind ∈ {status, token, done, error}。供 SSE 端点逐块推送 (进度 + 逐字报告)。
    """

    yield ("status", f"正在检索「{topic}」相关文献…")
    papers = []
    try:
        papers = await _fast_search(topic)
    except Exception as e:
        logger.warning(f"Search failed: {e}")

    yield ("status", f"检索到 {len(papers)} 篇，正在综合分析研究空白…")
    meta = [p.model_dump() for p in papers[:top_k]] if papers else []
    try:
        gen = await _synthesize(topic, meta, stream=True)
        async for token in gen:
            yield ("token", token)
    except Exception as e:
        logger.exception(f"Synthesis failed: {e}")
        yield ("error", "分析失败，请稍后重试")
        return
    yield ("token", _references_md(meta))
    yield ("done", len(papers))


async def _main(topic: str):
    res = await analyze_gap(topic)
    print(f"选题: {res['topic']}  ·  检索 {res['n_papers']} 篇\n")
    print(res["markdown"])


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    t = sys.argv[1] if len(sys.argv) > 1 else "检索增强生成RAG在医学问答中的应用"
    asyncio.run(_main(t))
