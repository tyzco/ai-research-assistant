"""产品模块 07 · 审稿 / 验证 / 对比
引用链追踪（Semantic Scholar 引用图）+ 论文关键信息核查。

演示:
    python demo/demo_07_review.py [paper_id]
"""
import sys

import common


async def main():
    st = common.load_state()
    pid = sys.argv[1] if len(sys.argv) > 1 else ""
    if not pid:
        papers = st.get("papers") or []
        # 挑一篇真论文: 有 DOI 且非占位条目 (标题不以 [ 开头)
        pid = next(
            (
                p.get("paper_id")
                for p in papers
                if p.get("doi") and not str(p.get("title", "")).startswith("[")
            ),
            papers[0].get("paper_id") if papers else "",
        )
        if not pid:
            print("  无论文可追踪。先运行: python demo/demo_01_search.py [或传 paper_id]")
            return

    common.banner(
        "产品模块 07 · 审稿 / 验证 / 对比",
        "引用链追踪 + 关键信息核查",
    )
    title = next((p.get("title", "") for p in (st.get("papers") or []) if p.get("paper_id") == pid), pid)
    common.section("论文信息")
    common.show("paper_id", pid)
    common.show("title", title[:90])

    common.section("引用链追踪（Semantic Scholar 引用图）")
    from integrations import expand_citation_chain

    res = await common.guard(
        lambda: expand_citation_chain(pid, direction="both", depth=1),
        "引用链（Semantic Scholar 不可达时跳过）",
    )
    if res:
        for node in (res.get("nodes") or [])[:8]:
            common.show("节点", str(node.get("title", ""))[:64] + " [" + str(node.get("year", "")) + "]")

    common.save_artifact(
        "07_review.md",
        "# 模块 07 · 审稿 / 验证 / 对比\n\n- 论文: " + title + "\n- 引用链: " + str(res)[:2000],
    )
    print("\n  （模块 07 为验证/对比能力入口：同行评审、伦理审查、格式校验见产品模块 05 下游层）")


if __name__ == "__main__":
    common.run(main)
