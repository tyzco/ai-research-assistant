"""产品模块 05 · 科研团队（13 Agent 流水线）
上游调研层(3) → 中游产出层(6) → 下游智控层(4)，3 个人工审批闸门，双轨机制。

默认展示团队架构（快速）；加 --full 跑真实全流程（演示自动通过人工闸门，耗时较长）。

演示:
    python demo/demo_05_team.py                 # 架构设计展示
    python demo/demo_05_team.py --full "调研RAG进展"  # 真实全流程
"""
import sys

import common

LAYERS = ["upstream", "midstream", "downstream"]
LAYER_CN = {
    "upstream": "上游调研层（选题 / 检索 / 筛选）",
    "midstream": "中游产出层（综述 / 引言 / 方法 / 实验 / 图表 / 参考文献）",
    "downstream": "下游智控层（审稿 / 格式 / 伦理 / 定稿）",
}


def show_design():
    common.section("团队架构（13 Agent · 3 层 · 3 闸门）")
    from agent.research_team import ResearchTeam

    t = ResearchTeam()
    for layer in LAYERS:
        agents = t.get_agents_by_layer(layer)
        print("  " + LAYER_CN.get(layer, layer))
        print("    " + " → ".join(agents))
    print("\n  人工审批闸门: 选题方向 → 核心文献入选 → 最终结论定稿")
    print("  双轨机制: Agent 自动执行 + 人工关键节点审批（防幻觉结论）")
    common.save_artifact(
        "05_team_design.md",
        "# 模块 05 · 科研团队架构\n\n"
        + "\n\n".join(
            "## " + LAYER_CN.get(l, l) + "\n\n" + " → ".join(
                ResearchTeam().get_agents_by_layer(l)
            )
            for l in LAYERS
        )
        + "\n\n## 人工审批闸门\n\n选题方向 → 核心文献入选 → 最终结论定稿",
    )


async def run_full(task: str):
    common.section("真实全流程（自动通过人工闸门）")
    from agent.research_team import ResearchTeam

    team = ResearchTeam()
    common.show("run_id", team.run_id)
    res = await team.execute(task)
    rounds = 0
    while rounds < 8 and (
        res.get("request") or res.get("paused")
        or res.get("status") in ("waiting_approval", "paused", "pending")
    ):
        rounds += 1
        common.ok("自动通过人工审批闸门 #" + str(rounds))
        res = await team.approve_and_continue()
    print("\n" + str(res.get("result", res))[:600])
    common.save_artifact("05_team_run.md", "# 模块 05 · 团队运行\n\n" + str(res)[:3000])


async def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    task = args[0] if args else "调研检索增强生成RAG的最新研究进展并输出综述"

    common.banner(
        "产品模块 05 · 科研团队（多 Agent 流水线）",
        "13 Agent 三层流水线 + 3 人工审批闸门",
    )
    show_design()
    if full:
        await run_full(task)
    else:
        print("\n  提示: 加 --full 运行真实全流程（多轮 LLM，耗时 2-5 分钟）")
    common.next_hint("python demo/demo_06_writing.py", "论文写作与多格式导出")


if __name__ == "__main__":
    common.run(main)
