"""产品模块 11 · 自我进化演示 (loop-engineering)
P1 反思重试 / P2 反馈教训 / P3 技能记忆 三环闭环。

演示:
    python demo/demo_11_self_evolution.py
"""
import sys

import common


def demo_p2():
    """反馈教训: 低分反馈 → 提炼「易失败教训」→ 注入系统提示。"""
    common.section("P2 反馈驱动调整（避免失败）")
    from database import get_db

    db = get_db()
    # 插入演示低分反馈
    db.record_feedback("Transformer原理是什么", "demo", 1, "回答太简短, 没有机制解释")
    db.record_feedback("对比RAG和微调", "demo", 1, "没有引用, 结论不扎实")
    db.record_feedback("注意力机制有哪些", "demo", 1, "遗漏多头注意力, 覆盖不全")

    from feedback_lessons import get_failure_lessons

    lessons = get_failure_lessons(force=True)
    if lessons:
        common.ok("低分反馈 → LLM 提炼教训:")
        for l in lessons:
            print("  - " + l)
    else:
        print("  （蒸馏不可用，检查网络）")

    # 验证注入系统提示
    try:
        from agent.pi_agent import PiAgent

        a = PiAgent(goal="测试")
        sp = a._build_system_prompt()
        if "历史反馈教训" in sp:
            common.ok("教训已注入 Agent 系统提示")
    except Exception as e:
        print("  （提示注入检查跳过: " + str(e)[:50] + "）")
    return lessons


def demo_p3():
    """技能记忆: 成功任务 → 记录做法 → 相似任务复用。"""
    common.section("P3 持久技能记忆（复用成功）")
    from agent_skills import record_success, get_similar_strategies

    record_success("调研RAG检索增强生成的现状并输出综述", ["search_papers", "agent_ask"], "已检索并综合 RAG 综述")
    record_success("用一句话介绍Transformer", [], "直接回答成功")
    common.ok("已记录 2 个成功任务做法")

    sims = get_similar_strategies("调研检索增强生成RAG的论文并总结")
    if sims:
        common.ok("相似任务匹配到经验:")
        for s in sims:
            print("  - 「" + str(s.get("task", ""))[:40] + "」→ 工具 [" + str(s.get("tools_used", "")) + "]")
    else:
        print("  （无匹配）")

    try:
        from agent.pi_agent import PiAgent

        a = PiAgent(goal="RAG调研")
        sp = a._build_system_prompt("", "调研RAG")
        if "过往相似任务" in sp:
            common.ok("相似经验已注入系统提示")
    except Exception as e:
        print("  （提示注入检查跳过: " + str(e)[:50] + "）")


def demo_p1():
    """反思重试: 展示机制启用 + 直答批判路径。"""
    common.section("P1 反思重试（当次改进）")
    from agent.pi_agent import PiAgent

    a = PiAgent(goal="测试", max_steps=2, max_time_seconds=60)
    common.ok("反思环启用: " + str(a.enable_reflexion))
    print("  机制: 直答后轻量自批判 → PASS 交付 / 非 PASS 带批判修订一轮（有界）")
    print("  说明: 演示任务已在上方 agent 运行中体现（成功则 PASS，不需修订）")


def main():
    common.banner(
        "产品模块 11 · 自我进化演示",
        "P1 反思重试 + P2 反馈教训 + P3 技能记忆 三环闭环",
    )
    demo_p2()
    demo_p3()
    demo_p1()

    md = (
        "# 模块 11 · 自我进化\n\n"
        "- P1 反思重试: 生成→自批判→修订（有界）\n"
        "- P2 反馈教训: 低分反馈→教训注入（避免失败）\n"
        "- P3 技能记忆: 成功任务→策略注入（复用成功）\n"
        "- 原则: 数据驱动, 人类把关, Agent 不改写自己\n"
    )
    common.save_artifact("11_self_evolution.md", md)


if __name__ == "__main__":
    common.run(main)
