"""产品模块 10 · 初稿审稿与批注
模拟同行评审: 5 维评分 + 总评 + 优点/不足 + 逐条批注 + 待澄清问题。

演示:
    python demo/demo_10_draft_review.py                # 用内置达标稿演示
    python demo/demo_10_draft_review.py --draft 初稿.md # 审给定初稿
"""
import argparse
from pathlib import Path

import common
from draft_review import DIMENSIONS, review_draft, _fmt

_SAMPLE = (
    "# 基于检索增强生成的医学问答系统初探\n\n"
    "## 摘要\n本文提出多源融合的 RAG 医学问答框架。\n\n"
    "## 引言\n医学问答需高准确与可溯源。\n\n"
    "## 相关工作\nLewis 等（2020）提出 RAG。\n\n"
    "## 方法\n三阶段框架：多源检索、增强、生成。\n\n"
    "## 结论与局限\n局限：实验未执行。\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="", help="审给定初稿文件")
    args = ap.parse_args()

    common.banner(
        "产品模块 10 · 初稿审稿与批注",
        "模拟同行评审：5 维评分 + 逐条批注 + 待澄清问题",
    )

    common.section("审稿维度")
    for d in DIMENSIONS:
        print("  - " + d)

    draft = Path(args.draft).read_text(encoding="utf-8") if args.draft else _SAMPLE
    report = review_draft(draft, "RAG医学问答")
    common.section("审稿报告")
    print("\n" + _fmt(report))

    common.save_artifact("10_draft_review.md", "# 模块 10 · 初稿审稿与批注\n\n" + _fmt(report))


if __name__ == "__main__":
    common.run(main)
