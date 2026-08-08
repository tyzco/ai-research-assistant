"""产品模块 08 · 初稿质量清单（选题→初稿 验收线）
6 项质量自检 → 判定 pass / revise。「货真价实初稿」= 研究问题明确 / 空白定位有据 /
结构完整 / 引用真实 / 诚实性 / 可续写。任何一项不过 → revise，不宣告完成。

确定性演示（快、可靠）:
    python demo/demo_08_draft_quality.py                 # 达标稿 vs 骨架稿 双案例
    python demo/demo_08_draft_quality.py --draft 初稿.md  # 检查给定初稿
    python demo/demo_08_draft_quality.py --agent 选题     # 真实 Agent 写稿后自检（较慢）
"""
import argparse
import asyncio
import sys
from pathlib import Path

import common
from draft_quality import DRAFT_CHECKLIST, check_draft

_GOOD_DRAFT = """# 基于检索增强生成的医学问答系统初探

## 摘要
医学问答面临知识时效与专业覆盖率不足的问题。本文调研检索增强生成（RAG）在医学问答中的应用现状，
梳理检索、增强、生成三阶段的主流方法与公开数据集，指出现有工作在多源异构医学知识融合上仍有空白，
并规划了改进思路与实验。

## 引言
医学领域的专业问答对准确性与可溯源要求高，通用大模型存在幻觉与知识过时问题。检索增强生成通过外接
知识库缓解该问题（Lewis et al., 2020）。但在垂直医学场景，如何融合教科书、临床指南、期刊摘要等多源
异构知识仍缺乏系统研究。本文聚焦该空白，目标是验证多源融合策略对医学问答准确率的提升。

## 相关工作
RAG 通用框架由 Lewis 等（2020）提出，检索器+生成器的范式成为主流；后续工作（Izacard & Grave, 2021）
改进了稠密检索融合。在医学领域，Singhal 等（2023）评估了 PaLM 在医学问答上的表现，指出外接知识能
显著降低幻觉。就检索语料而言，现有工作多以单一语料为主（Singhal et al., 2023），多源异构医学知识的
融合与消歧尚未被系统解决——该空白判断基于上述文献的语料范围，需在进一步检索后确认，存在不确定性。

## 方法
本文提出三阶段框架：1) 多源检索——对教科书、临床指南、期刊摘要分别建索引并融合排序；2) 增强——
引入来源权重与冲突消歧；3) 生成——基于检索证据约束生成并强制标注引用。

## 实验计划
使用 MedQA 与自建医学子集评测，基线为单源 RAG 与纯 LLM，指标为准确率、引用可溯源率、幻觉率。

## 结论与局限
预期多源融合能提升准确率与可溯源率。局限：本文为初稿，实验尚未执行，结论为假设；医学知识存在地域
差异，评测集覆盖面有限。

## 参考文献
Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models. arXiv.
Singhal, K., et al. (2023). Large Language Models Encode Clinical Knowledge. Nature.
"""

_WEAK_DRAFT = """# RAG在医学问答中的应用

摘要。现有方法有局限。

## 相关工作
[Title, 2020] 提出 A。[Title, 2022] 改进 B。

## 方法
我们提出三模块框架：检索、增强、生成。
"""


def _render(report: dict, label: str) -> None:
    common.section(label)
    for r in report["items"]:
        mark = "[OK] " if r["pass"] else "[X]  "
        print("  " + mark + r["label"] + ": " + r["reason"])
    print(
        "  → 通过 " + str(report["passed"]) + "/" + str(report["total"])
        + " · 判定: " + report["verdict"]
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="", help="检查给定初稿文件")
    ap.add_argument("--agent", nargs="?", const="RAG在垂直领域的应用初探", default="", help="真实 Agent 写稿后自检")
    args = ap.parse_args()

    common.banner(
        "产品模块 08 · 初稿质量清单",
        "选题→初稿 验收线：6 项自检 → pass / revise",
    )

    common.section("清单规格（货真价实初稿的验收标准）")
    for i in DRAFT_CHECKLIST:
        print("  [" + i["id"].ljust(10) + "] " + i["label"] + " — " + i["check"])

    if args.draft:
        draft = Path(args.draft).read_text(encoding="utf-8")
        _render(check_draft(draft, [], Path(args.draft).stem), "给定初稿自检")
        return

    if args.agent:
        common.section("Agent 撰写初稿")
        from agent.pi_agent import PiAgent

        task = (
            "针对选题「" + args.agent + "」写一篇论文初稿（Markdown）："
            "包含 摘要/引言/相关工作/方法/实验计划/结论与局限，每个论点标注 [Title, Year]，显式声明局限。"
        )
        common.step("Pi-Agent（paper_writing 技能）撰写中（研究+写作，约 1-3 分钟）")
        agent = PiAgent(goal=task[:80], max_steps=12, max_time_seconds=300, skills=["paper_writing"])
        res = await agent.run(task)
        draft = str(res.get("result", ""))
        print("\n" + (draft[:800] if draft else "（Agent 未产出有效初稿）"))
        if not draft or len(draft) < 100:
            print("  回退: 用内置达标稿演示清单判定")
            draft = _GOOD_DRAFT
        _render(check_draft(draft, [], args.agent), "Agent 初稿自检")
        return

    # 确定性双案例
    good = check_draft(_GOOD_DRAFT, [], "RAG医学问答")
    bad = check_draft(_WEAK_DRAFT, [], "RAG医学问答")
    _render(good, "案例 A · 达标初稿（预期 pass 或个别项待完善）")
    _render(bad, "案例 B · 骨架半成品（预期 revise）")
    if good["verdict"] == "pass":
        print("\n  说明：达标稿通过验收；骨架稿未达标，Agent 需修订后再交付。")


if __name__ == "__main__":
    common.run(main)
