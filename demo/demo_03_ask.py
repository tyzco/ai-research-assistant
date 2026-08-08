"""产品模块 03 · 私有知识库问答（RAG）
问题 → 指代消解 → 查询改写 → 自适应混合检索 → 级联重排 → 父子分块扩展 → 生成（带引用）。

演示:
    python demo/demo_03_ask.py "这些论文主要解决了什么问题？"
"""
import sys

import common

Q = sys.argv[1] if len(sys.argv) > 1 else "这些论文主要解决了什么问题？"


async def main():
    common.banner(
        "产品模块 03 · 私有知识库问答（RAG）",
        "消解 → 改写 → 混合检索 → 级联重排 → 父子扩展 → 带引用生成",
    )
    st = common.load_state()
    if not st.get("table"):
        print("  请先运行: python demo/demo_02_kb.py")
        return
    from qa_engine import ask_question

    common.section("RAG 问答（真实链路）")
    common.show("问题", Q)
    common.show("知识库", st["table"])
    common.step("检索 + 重排 + 生成")
    resp = await ask_question(st["table"], Q, memory_key=st["topic_id"])
    print("\n" + str(resp.answer))

    if resp.references:
        common.section("引用（可溯源）")
        for r in resp.references[:6]:
            common.show("论文", str(r.get("title", ""))[:64] + " [" + str(r.get("year", "")) + "]")

    md = (
        "# 模块 03 · 私有知识库问答\n\n**Q:** " + Q + "\n\n" + str(resp.answer)
        + "\n\n## 引用\n"
        + "\n".join("- " + str(r.get("title", "")) for r in resp.references)
    )
    common.save_artifact("03_ask.md", md)
    common.next_hint("python demo/demo_04_agent.py", "Agent 即服务：把任务交给智能体")


if __name__ == "__main__":
    common.run(main)
