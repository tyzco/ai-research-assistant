"""产品模块 09 · 研究空白分析（选题可行性）
回答「这个选题有没有人做过 / 可行性如何」→ 真实检索 → 空白报告。

演示:
    python demo/demo_09_gap_analysis.py "选题"
"""
import asyncio
import sys

import common


async def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "检索增强生成RAG在医学问答中的应用"
    common.banner(
        "产品模块 09 · 研究空白分析（选题可行性）",
        "「这个选题有人做过吗」→ 真实检索 → 空白报告 → 可行性",
    )
    from research_gap import analyze_gap

    common.section("真实检索 + LLM 综合")
    common.step("检索 12 学术源")
    res = await analyze_gap(topic)
    common.ok("检索 " + str(res["n_papers"]) + " 篇 → 综合生成空白报告")

    r = res.get("report", {})
    common.section("已有工作概览")
    print("  " + str(r.get("existing_summary", "")))

    common.section("已覆盖的子问题")
    for a in r.get("covered_areas", []) or []:
        print("  - " + a)

    common.section("研究空白（候选方向，锚定文献）")
    gaps = r.get("research_gaps", []) or []
    if not gaps:
        print("  （未识别到明确空白，需扩大检索 / 人工判断）")
    for g in gaps:
        print("  - " + str(g.get("gap", "")))
        print("    依据: " + str(g.get("evidence", "")) + " · 置信度: " + str(g.get("confidence", "")))

    f = r.get("feasibility", {}) or {}
    common.section("可行性")
    print("  - 数据: " + str(f.get("data", "")))
    print("  - 方法: " + str(f.get("method", "")))
    print("  - 风险: " + str(f.get("risk", "")))

    common.section("不确定性声明")
    print("  " + str(r.get("uncertainty", "")))

    md = (
        "# 模块 09 · 研究空白分析\n\n- 选题: " + topic + "\n- 检索 " + str(res["n_papers"]) + " 篇\n\n"
        + str(r.get("existing_summary", "")) + "\n\n## 研究空白\n"
        + "\n".join("- " + str(g.get("gap", "")) + " (" + str(g.get("confidence", "")) + ")" for g in gaps)
    )
    common.save_artifact("09_gap.md", md)


if __name__ == "__main__":
    common.run(main)
