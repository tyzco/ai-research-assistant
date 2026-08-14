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
import json
import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

_SYNTH_PROMPT = """你是学术研究选题顾问。基于以下【检索到的真实文献】，对选题「{topic}」做研究空白与可行性分析。
只输出 JSON，不要解释。要求：每个空白判断必须基于检索到的文献，并标注不确定性；宁可说"未覆盖/不确定"，也不编造空白。

输出格式:
{{
  "existing_summary": "已有工作概览（1-2 句，基于真实文献）",
  "covered_areas": ["已充分研究的子问题", ...],
  "research_gaps": [
    {{"gap": "候选空白方向", "evidence": "基于哪些文献/未覆盖哪方面", "confidence": "高|中|低"}}
  ],
  "feasibility": {{"data": "数据可得性", "method": "方法成熟度", "risk": "主要风险"}},
  "uncertainty": "检索局限与不确定性声明（如: 本分析基于检索到的 N 篇, 需扩大检索/加人工复核确认）"
}}

【检索到的文献】
{papers}

【选题】{topic}
"""


def _synth(topic: str, papers: list, n_papers: int) -> dict:
    lines = []
    for i, p in enumerate(papers[:15], 1):
        lines.append(
            f"[{i}] {p.get('title','')} ({p.get('year','')}) · {p.get('source','')}"
            + (
                f"\n    abstract: {(p.get('abstract') or '')[:200]}"
                if p.get("abstract")
                else ""
            )
        )
    papers_txt = "\n".join(lines) or "（检索无结果）"
    prompt = _SYNTH_PROMPT.format(topic=topic, papers=papers_txt, n_papers=n_papers)
    try:
        from openai import OpenAI

        from config import (
            DEEPSEEK_API_KEY,
            DEEPSEEK_BASE_URL,
            DEEPSEEK_FALLBACK_API_KEY,
            LLM_STRATEGY_MAX_TOKENS,
            RESEARCH_GAP_MODEL,
        )

        # 选题判断钩子用 pro 模型 + pro key (兜底 key 优先), flash 省钱留给后台/推送
        _gap_key = DEEPSEEK_FALLBACK_API_KEY or DEEPSEEK_API_KEY
        client = OpenAI(api_key=_gap_key, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=RESEARCH_GAP_MODEL,
            temperature=0.3,
            max_tokens=LLM_STRATEGY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
            timeout=60,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        if not isinstance(data, dict):
            raise ValueError("non-dict")
        return data
    except Exception as e:
        logger.warning(f"Gap synthesis failed: {e}")
        return {
            "existing_summary": "文献检索结果见上，LLM 综合不可用",
            "covered_areas": [],
            "research_gaps": [],
            "feasibility": {},
            "uncertainty": "LLM 综合失败，请人工判断",
        }


async def analyze_gap(topic: str, top_k: int = 15) -> dict:
    """选题 → 研究空白报告。返回 {topic, n_papers, papers, report}。

    检索失败时优雅降级: 返回空报告并标注, 不中断。
    """
    from search import search_papers_for_topic

    papers = []
    try:
        papers = await search_papers_for_topic(topic)
    except Exception as e:
        logger.warning(f"Search failed for gap analysis: {e}")

    meta = [p.model_dump() for p in papers[:top_k]] if papers else []
    report = _synth(topic, meta, len(papers))
    return {
        "topic": topic,
        "n_papers": len(papers),
        "papers": meta,
        "report": report,
    }


def _fmt(report: dict) -> str:
    r = report.get("report", {})
    lines = [
        "## 已有工作概览",
        r.get("existing_summary", ""),
        "\n## 已覆盖的子问题",
    ]
    lines += ["- " + a for a in r.get("covered_areas", [])] or ["- （无）"]
    lines.append("\n## 研究空白（候选方向）")
    for g in r.get("research_gaps", []):
        lines.append(
            f"- {g.get('gap','')} —— 依据: {g.get('evidence','')} · 置信度: {g.get('confidence','')}"
        )
    if not r.get("research_gaps"):
        lines.append("- （未识别到明确空白，需扩大检索/人工判断）")
    f = r.get("feasibility", {})
    lines += [
        "\n## 可行性",
        f"- 数据: {f.get('data','')}",
        f"- 方法: {f.get('method','')}",
        f"- 风险: {f.get('risk','')}",
        "\n## 不确定性声明",
        r.get("uncertainty", ""),
    ]
    return "\n".join(lines)


async def _main(topic: str):
    res = await analyze_gap(topic)
    print(f"选题: {res['topic']}  ·  检索 {res['n_papers']} 篇\n")
    print(_fmt(res))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    t = sys.argv[1] if len(sys.argv) > 1 else "检索增强生成RAG在医学问答中的应用"
    asyncio.run(_main(t))
