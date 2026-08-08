"""一键串联演示：检索 → 建库 → 问答 → Agent → 团队 → 写作 → 审稿。
各模块经 demo/output/state.json 状态衔接，验证产品模块可独立展示、可串联交付。

用法:
    python demo/run_all.py                # 默认研究方向
    python demo/run_all.py "多模态大模型"
"""
import subprocess
import sys

# (脚本, 是否接收研究方向作为参数)
STEPS = [
    ("demo_01_search.py", True),
    ("demo_09_gap_analysis.py", True),
    ("demo_02_kb.py", False),
    ("demo_03_ask.py", False),
    ("demo_04_agent.py", False),
    ("demo_05_team.py", False),
    ("demo_06_writing.py", False),
    ("demo_08_draft_quality.py", False),
    ("demo_10_draft_review.py", False),
    ("demo_11_self_evolution.py", False),
    ("demo_12_proactive.py", False),
    ("demo_07_review.py", False),
]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "大语言模型代码漏洞自动修复"
    for script, take_query in STEPS:
        print("\n" + "#" * 72)
        print("# 运行: " + script)
        print("#" * 72)
        cmd = [sys.executable, "demo/" + script]
        if take_query:
            cmd.append(query)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("\n[run_all] " + script + " 失败，终止。")
            sys.exit(1)
    print("\n全部 12 个产品模块演示完成。产物见 demo/output/（可直接出示）")


if __name__ == "__main__":
    main()
