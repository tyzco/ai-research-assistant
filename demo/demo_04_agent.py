"""产品模块 04 · Agent 即服务（Pi-Agent）
任务 → 技能自动路由 → 工具链（检索/建库/问答）→ Hook 生命周期 → 人工确认门 → 结果。

演示:
    python demo/demo_04_agent.py "调研RAG研究现状，输出结构化综述"
"""
import sys

import common

TASK = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "调研检索增强生成RAG的研究现状，用结构化 Markdown 输出综述"
)


async def main():
    common.banner(
        "产品模块 04 · Agent 即服务（Pi-Agent）",
        "技能路由 → 工具调用 → Hook 生命周期 → 人工确认门",
    )
    from agent.pi_agent import PiAgent

    common.section("Pi-Agent 运行")
    common.show("任务", TASK)
    common.step("技能自动路由 → 工具链执行")
    agent = PiAgent(goal=TASK[:80], max_steps=6, max_time_seconds=180)
    res = await agent.run(TASK)

    print("\n" + str(res.get("result", ""))[:900])

    trace = res.get("trace")
    if trace:
        common.section("执行轨迹（每步工具调用）")
        for s in (trace.get("steps") or [])[:12]:
            t = s.get("tool", "llm")
            mark = "[OK]" if s.get("success") else "[X]"
            print(
                "  " + mark + " #" + str(s.get("turn", "?"))
                + " " + str(t) + " (" + str(s.get("elapsed_ms", 0))
                + "ms / " + str(s.get("tokens", 0)) + " tok)"
            )
        common.show("总轮次", trace.get("turns", ""))
        common.show("总耗时", str(trace.get("elapsed", "")) + "s")

    common.save_artifact(
        "04_agent.md",
        "# 模块 04 · Agent 即服务\n\n**任务:** " + TASK + "\n\n" + str(res.get("result", "")),
    )
    common.next_hint("python demo/demo_05_team.py", "13 Agent 科研团队全流程")


if __name__ == "__main__":
    common.run(main)
