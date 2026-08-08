"""初稿质量清单 —— 「选题 → 初稿」交付的验收线。

「货真价实的初稿」= 符合论文最基本要求：研究问题明确 / 空白定位有据 /
结构完整 / 引用真实 / 诚实性 / 可续写。Agent 完成初稿前逐项自检，
任何一项不过 → 判定 revise，不宣告完成。

判定方式：
  - structure / citations  用规则快判（章节标题 + 引用标记/参考文献列表）
  - question / gap / honesty / completeness  用 DeepSeek 直判
    （与 evaluate_rag_standards 同模式：同步客户端 + thinking disabled，
     避免 ragas LLM judge 与 DeepSeek 输出格式不兼容的问题）

自测: python draft_quality.py
"""

import json
import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

# ===== 清单规格（可验收标准）=====
DRAFT_CHECKLIST = [
    {"id": "question", "label": "研究问题明确", "check": "有明确、具体、有边界的研究问题"},
    {"id": "gap", "label": "空白定位", "check": "基于引用的相关工作定位空白，诚实标注不确定"},
    {"id": "structure", "label": "结构完整", "check": "摘要/引言/相关工作/方法/实验计划/结论齐全"},
    {"id": "citations", "label": "引用真实", "check": "每个论点带真实引用，无凭空编造"},
    {"id": "honesty", "label": "诚实性", "check": "局限与不确定性显式声明，不夸大"},
    {"id": "completeness", "label": "可续写", "check": "是能继续打磨的完整初稿，而非半成品"},
]

RULE_ITEMS = {"structure", "citations"}
LLM_ITEMS = {"question", "gap", "honesty", "completeness"}

_SECTIONS = ["摘要", "引言", "相关", "方法", "实验", "结论"]


# ===== 规则判定（快、零成本、可靠）=====

def _rule_check(draft: str, references: list) -> dict:
    d = (draft or "").lower()

    found = [s for s in _SECTIONS if s in d]
    missing = [s for s in _SECTIONS if s not in found]
    structure_pass = len(found) >= 4
    results = {
        "structure": {
            "pass": structure_pass,
            "reason": (
                "命中章节: " + "、".join(found)
                if structure_pass
                else "仅命中: " + "、".join(found) + "；缺: " + "、".join(missing)
            ),
        }
    }

    has_ref_list = bool(references) or ("参考文献" in d) or ("references" in d)
    # 行内引用: [n] / [Title, Year] / (Author, Year) / （Author, Year）全角括号都认
    has_inline = bool(
        __import__("re").search(
            r"\[\d+\]|\[论文:|\[[A-Za-z][^\]\[]{2,40}, 20\d\d\]"
            r"|\([A-Za-z][^()]{1,40}, 20\d\d\)"
            r"|（[A-Za-z][^（）]{1,40}, 20\d\d）",
            draft or "",
        )
    )
    citations_pass = has_ref_list and has_inline
    results["citations"] = {
        "pass": citations_pass,
        "reason": (
            "有参考文献列表 + 行内引用标记"
            if citations_pass
            else "缺: " + ("参考文献列表" if not has_ref_list else "") + ("行内引用标记" if not has_inline else "")
        ),
    }
    return results


# ===== LLM 判定（语义项）=====

def _llm_check(draft: str, references: list, goal: str = "") -> dict:
    items = [i for i in DRAFT_CHECKLIST if i["id"] in LLM_ITEMS]
    refs_txt = "\n".join(
        f"- {r.get('title','')}" + (f" [{r.get('year')}]" if r.get("year") else "")
        for r in (references or [])[:20]
    )
    prompt = (
        "你是论文初稿质量评审。逐项判定以下 4 项是否达标，只输出 JSON：\n"
        + json.dumps(
            [
                {"id": i["id"], "label": i["label"], "check": i["check"]}
                for i in items
            ],
            ensure_ascii=False,
        )
        + "\n\n目标: " + goal[:300] + "\n\n参考文献:\n" + (refs_txt or "（无）")
        + "\n\n初稿:\n" + (draft or "")[:4000]
        + "\n\n输出格式: {\"question\": {\"pass\": true, \"reason\": \"...\"}, ...} "
          "reason 一句话，中文。判定从严：无法确认引用真实时 citations 之外的 honesty 也要反映不确定。"
    )
    try:
        from openai import OpenAI

        from config import (
            DEEPSEEK_API_KEY,
            DEEPSEEK_BASE_URL,
            DEEPSEEK_MODEL,
            LLM_QUICK_MAX_TOKENS,
        )

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            max_tokens=min(LLM_QUICK_MAX_TOKENS, 600),
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
            timeout=30,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {
            iid: {
                "pass": bool(v.get("pass", False)),
                "reason": str(v.get("reason", ""))[:120],
            }
            for iid, v in data.items()
            if isinstance(v, dict)
        }
    except Exception as e:
        logger.warning(f"Draft LLM check failed: {e}")
        return {}


# ===== 主入口 =====

def check_draft(draft: str, references: list, goal: str = "") -> dict:
    """初稿质量清单自检。返回 {items, passed, total, verdict}。

    verdict: "pass" = 全部达标可交付; "revise" = 有未达标项, 需修订;
             "unknown" = 语义项判定不可用, 需人工核对。
    """
    rule = _rule_check(draft, references)
    llm = _llm_check(draft, references, goal)

    results = []
    for item in DRAFT_CHECKLIST:
        if item["id"] in RULE_ITEMS:
            r = rule[item["id"]]
            results.append({**item, "pass": r["pass"], "reason": r["reason"]})
        else:
            r = llm.get(item["id"])
            if r is None:
                results.append(
                    {**item, "pass": False, "reason": "自动判定不可用，请人工核对"}
                )
            else:
                results.append({**item, "pass": r["pass"], "reason": r["reason"]})

    passed = sum(1 for r in results if r["pass"])
    unknown = sum(1 for r in results if r["reason"] == "自动判定不可用，请人工核对")
    if unknown:
        verdict = "unknown"
    else:
        verdict = "pass" if passed == len(results) else "revise"
    return {"items": results, "passed": passed, "total": len(results), "verdict": verdict}


# ===== 自测 =====

def _selftest() -> None:
    good = (
        "# 标题\n\n## 摘要\n本工作解决...\n\n## 引言\n背景与动机...\n\n"
        "## 相关工作\n[1] 描述了 A。[2] 描述了 B。\n\n## 方法\n我们提出...\n\n"
        "## 实验计划\n我们将在...上评测。\n\n## 结论与局限\n结论... 局限...\n\n"
        "## 参考文献\n[1] Smith, 2020. ...\n[2] 张三, 2021. ...\n"
    )
    bad = "这是一段没有章节也没有引用的描述。"
    print("=== 达标初稿 ===")
    for r in check_draft(good, [{"title": "x", "year": 2020}], "示例")["items"]:
        print(f"  {'[OK]' if r['pass'] else '[X] '} {r['label']}: {r['reason']}")
    print("\n=== 不达标初稿 ===")
    res = check_draft(bad, [], "示例")
    print("  verdict =", res["verdict"], f"({res['passed']}/{res['total']})")
    for r in res["items"]:
        print(f"  {'[OK]' if r['pass'] else '[X] '} {r['label']}: {r['reason']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
