"""产品演示总纲 (Pitch) — 「选题 → 初稿」完整故事线。

一键演示核心产品价值, 每步带旁白 (产品定位 + 变现指向), 输出完整演示稿。
用于拉投资 / 项目展示 / 吸引协作。

用法:
    python demo/pitch.py "研究方向"
"""
import subprocess
import sys

import common

# (模块, 是否接收研究方向参数, 旁白)
STEPS = [
    (
        "demo_09_gap_analysis.py", True,
        "① 研究空白分析 —— 输入选题，回答「这个选题有没有人做过」，给出可行性判断（锚定真实文献，诚实标注不确定性）",
    ),
    (
        "demo_01_search.py", True,
        "② 论文检索 —— 12 学术源并行，主题相关度重排，浮出关键论文",
    ),
    (
        "demo_02_kb.py", False,
        "③ 私有知识库 —— 用户自供论文构建双层索引（摘要+全文），隐私友好，合法护城河",
    ),
    (
        "demo_03_ask.py", False,
        "④ 知识库问答 —— RAG 基于证据回答，可溯源引用，拒绝幻觉",
    ),
    (
        "demo_04_agent.py", False,
        "⑤ Agent 即服务 —— 多步工具调用，技能自动路由，人工确认门（这是变现核心：按结果卖）",
    ),
    (
        "demo_08_draft_quality.py", False,
        "⑥ 初稿质量验收 —— 6 项自检（研究问题/空白/结构/引用/诚实性/可续写），货真价实的初稿",
    ),
    (
        "demo_10_draft_review.py", False,
        "⑦ 审稿批注 —— 模拟同行评审 5 维评分 + 逐条批注 + 待澄清问题",
    ),
    (
        "demo_11_self_evolution.py", False,
        "⑧ 自我进化 —— 反思重试 + 反馈教训 + 技能记忆，系统越用越聪明",
    ),
    (
        "demo_12_proactive.py", False,
        "⑨ 主动路由 —— 预判用户旅程下一步、一键执行、接受度学习，知用户递到跟前",
    ),
]

_SUMMARY = """
==========================================================
  产品价值总结
==========================================================
  定位: C 端个人学术工具 — 帮用户完成「选题 → 初稿」
  交付: 货真价实、符合论文基本要求的初稿
  Agent 即服务: 免费额度体验 → 按量付费 (完成一篇退场)
  差异化:
    - 私有知识库 (用户数据护城河, 隐私友好)
    - 研究空白分析 (「有没有人做过」, 锚定文献)
    - 初稿质量清单 + 审稿批注 (货真价实验收线)
    - 自我进化闭环 (反思 + 反馈教训 + 技能记忆)
    - 主动路由 (预判下一步 + 一键执行 + 接受度学习)
    - 合法边界 (仅 OA + 用户上传, 无盗版)
  演示产物: demo/output/*.md
==========================================================
"""


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "检索增强生成RAG在医学问答中的应用"
    common.banner(
        "PaperLink 产品演示总纲",
        "「选题 → 初稿」完整故事线 · 8 步 · 真实调用",
    )
    transcript = ["# PaperLink 产品演示", "", "研究方向: " + query, ""]

    for script, take_query, narration in STEPS:
        print("\n" + "─" * 60)
        print(narration)
        print("─" * 60)
        transcript.append("## " + narration)
        cmd = [sys.executable, "demo/" + script]
        if take_query:
            cmd.append(query)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("\n[pitch] " + script + " 失败，终止。")
            sys.exit(1)
        # 收集该模块产物进演示稿
        md = common.OUT / script.replace(".py", ".md").replace("demo_", "")
        if md.exists():
            transcript.append("```\n" + md.read_text(encoding="utf-8")[:600] + "\n```")

    print(_SUMMARY)
    transcript.append(_SUMMARY)
    transcript.append("\n→ 完整演示产物见 demo/output/（可直接出示）")
    out = common.OUT / "pitch_transcript.md"
    out.write_text("\n".join(transcript), encoding="utf-8")
    print("\n完整演示稿已保存: " + out.relative_to(common.ROOT).as_posix())


if __name__ == "__main__":
    common.run(main)
