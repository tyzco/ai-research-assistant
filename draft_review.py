"""初稿审稿与批注 —— 「帮忙审查和批注」环节。

对用户初稿做模拟同行评审: 5 维评分 + 总评 + 优点/不足 + 逐条批注 + 待澄清问题。
维度复用 research_team 的 peer_reviewer (技术正确性/实验说服力/贡献清晰度/写作质量/可复现性)。

自测: python draft_review.py
"""

import json
import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

DIMENSIONS = ["技术正确性", "实验说服力", "贡献清晰度", "写作质量", "可复现性"]

_REVIEW_PROMPT = """你是模拟同行审稿人，以顶会审稿标准审查论文初稿。
只输出 JSON，不要解释。

## 审稿维度 (1-5 分)
1. 技术正确性: 方法是否有理论支撑? 推导是否严密?
2. 实验说服力: 实验设计是否合理? 对比基线是否充分?
3. 贡献清晰度: 核心贡献是否明确? 与已有工作的区别是否清楚?
4. 写作质量: 结构是否清晰? 语言是否准确?
5. 可复现性: 是否提供足够细节供他人复现?

## 输出格式
{{
  "overall": "Accept | Borderline | Revise",
  "scores": {{"技术正确性": 0, "实验说服力": 0, "贡献清晰度": 0, "写作质量": 0, "可复现性": 0}},
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "comments": [{{"location": "哪一节/哪句话", "note": "具体批注与修改建议"}}],
  "questions": ["待作者澄清的问题"]
}}

【研究目标】(可选) {goal}

【初稿】
{draft}
"""


def review_draft(draft: str, goal: str = "") -> dict:
    """初稿审稿。返回 {overall, scores, strengths, weaknesses, comments, questions}。
    LLM 失败时降级返回空结构 + note。"""
    prompt = _REVIEW_PROMPT.format(goal=goal[:300], draft=(draft or "")[:6000])
    fallback = {
        "overall": "Unknown",
        "scores": {d: 0 for d in DIMENSIONS},
        "strengths": [],
        "weaknesses": [],
        "comments": [],
        "questions": [],
        "note": "LLM 审稿不可用",
    }
    try:
        from openai import OpenAI

        from config import (
            DEEPSEEK_API_KEY,
            DEEPSEEK_BASE_URL,
            DEEPSEEK_MODEL,
            LLM_STRATEGY_MAX_TOKENS,
        )

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.2,
            max_tokens=LLM_STRATEGY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
            timeout=60,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        if not isinstance(data, dict):
            raise ValueError("non-dict")
        return data
    except Exception as e:
        logger.warning(f"Draft review failed: {e}")
        fallback["note"] = "LLM 审稿不可用: " + str(e)[:80]
        return fallback


def _fmt(report: dict) -> str:
    lines = ["## 总评: " + str(report.get("overall", "Unknown"))]
    s = report.get("scores", {}) or {}
    lines.append("## 评分")
    for d in DIMENSIONS:
        lines.append(f"- {d}: {s.get(d, '?')}/5")
    lines.append("## 优点")
    lines += ["- " + x for x in report.get("strengths", []) or []] or ["- （无）"]
    lines.append("## 不足")
    lines += ["- " + x for x in report.get("weaknesses", []) or []] or ["- （无）"]
    lines.append("## 逐条批注")
    for c in report.get("comments", []) or []:
        lines.append(f"- [{c.get('location','')}] {c.get('note','')}")
    if not report.get("comments"):
        lines.append("- （无）")
    lines.append("## 待澄清问题")
    lines += ["- " + q for q in report.get("questions", []) or []] or ["- （无）"]
    if report.get("note"):
        lines.append("## 说明")
        lines.append(report["note"])
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = (
        "# 测试初稿\n\n## 摘要\n提出一种方法。\n\n## 引言\n背景。\n\n"
        "## 方法\n我们做了 X。\n\n## 结论\n结论。\n"
    )
    print(_fmt(review_draft(sample)))
