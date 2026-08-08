"""产品模块 12 · 主动路由 (预判 → 执行 → 学习)

展示「知用户、递到跟前、越用越懂」的路由闭环:
  v1 阶段预判: 每个旅程阶段给出候选下一步 (默认顺序)
  v2 一键执行: 建议可点击直接执行 (见前端 suggestBar)
  v4 接受度学习: 用户点「执行」越多, 该建议越优先推送

演示:
    python demo/demo_12_proactive.py
"""
import os

import common

STAGES = {
    "idle": [{"action": "new_topic", "suggest": "输入研究方向，先做研究空白分析"}],
    "created": [
        {"action": "search", "suggest": "查看论文卡片"},
        {"action": "build_kb", "suggest": "下载 OA 建库"},
    ],
    "searched": [
        {"action": "build_kb", "suggest": "下载 OA 全文构建知识库"},
        {"action": "ask", "suggest": "先基于摘要提问"},
    ],
    "kb_ready": [
        {"action": "ask", "suggest": "直接提问，基于知识库回答"},
        {"action": "write", "suggest": "让 Agent 写综述/初稿"},
    ],
    "writing": [
        {"action": "write", "suggest": "让 Agent 写初稿，做完质量验收"},
        {"action": "ask", "suggest": "继续追问细节"},
    ],
}


def main():
    common.banner(
        "产品模块 12 · 主动路由",
        "预判 → 执行 → 学习（知用户、递到跟前、越用越懂）",
    )
    from suggestions import get_stats, rank_candidates, record_accept

    common.section("v1 阶段预判（每个旅程阶段 → 候选下一步）")
    for stage, cands in STAGES.items():
        top = rank_candidates(stage, cands)[0]
        print(
            f"  {stage:10s} → 优先推「{top['action']}」: {top['suggest']}"
        )

    common.section("v4 接受度学习（记录 → 排序翻转）")
    print("  默认 kb_ready 优先 ask; 若用户连续接受 write:")
    for _ in range(3):
        record_accept("kb_ready", "write")
    record_accept("kb_ready", "ask")
    top = rank_candidates("kb_ready", STAGES["kb_ready"])[0]
    print(f"  记录 write×3 / ask×1 后 → kb_ready 优先「{top['action']}」（从 ask 翻转到 write）")

    common.section("学习统计（dev.html「主动路由学习」实时可见）")
    stats = get_stats()
    if stats:
        for stage, acts in stats.items():
            print(f"  {stage}: " + " · ".join(f"{a}×{c}" for a, c in acts.items()))

    # 清理演示数据 (不污染真实学习)
    if os.path.exists("data/suggestion_stats.json"):
        os.remove("data/suggestion_stats.json")
    common.ok("演示学习数据已清理")

    common.save_artifact(
        "12_proactive.md",
        "# 模块 12 · 主动路由\n\n"
        "- v1 阶段预判: 每阶段候选下一步\n"
        "- v2 一键执行: 建议条可点击执行\n"
        "- v4 接受度学习: 用户点执行越多越优先推\n"
        "- 演进: 预判 → 执行 → 记录 → 学习 → 更准的预判",
    )


if __name__ == "__main__":
    common.run(main)
