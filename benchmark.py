"""性能基准测试：检索延迟 + 语义命中率 + Token 估算。

用法:
    python benchmark.py                  # 自动选择 KB
    python benchmark.py --kb papers_xxx  # 指定表
    python benchmark.py --qa data/eval_qa.json --topk 5

DeepSeek 审查: 空表防护/CLI 参数(#3), 数据校验(#4), 命中率修正(#1),
CN-EN token 估算(#2), 批量嵌入(#6), JSON 报告(#9), 成本可配(#8)。
"""
import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np

import lancedb

from eval_common import est_tokens, load_eval_qa, pick_kb, semantic_recall
from knowledge_base import LANCEDB_DIR, _cosine_search, _embed_texts_sync

# 模型价格 ($/百万 token), 可用环境变量覆盖
MODEL_COST_PER_1M = float(os.getenv("MODEL_COST_PER_1M", "0.14"))


async def bench():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="", help="指定 LanceDB 表名 (默认自动选择)")
    ap.add_argument("--qa", default="data/eval_qa.json", help="测试集路径")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="仅评估前 N 题 (默认全部)")
    ap.add_argument("--report", default="data/benchmark_report.json")
    args = ap.parse_args()

    qa = load_eval_qa(args.qa, args.limit)
    db = lancedb.connect(str(LANCEDB_DIR))
    kb = pick_kb(args.kb)
    tbl = db.open_table(kb)
    rows = tbl.count_rows()
    print(f"KB: {kb} ({rows} rows) | 测试题: {len(qa)}\n")

    # 批量嵌入问题 (单次模型调用, 大幅提速 #6)
    questions = [str(i["question"])[:200] for i in qa]
    q_embs = _embed_texts_sync(questions)

    times, hits, tokens = [], [], []
    results = []
    print(f'{"#":>3s} {"Question":34s} {"Time":>7s} {"Hit":>6s}')
    total = len(qa)
    for idx, item in enumerate(qa):
        q = str(item["question"])[:200]
        gt = str(item.get("ground_truth", ""))
        qv = q_embs[idx]
        t0 = time.time()
        res = _cosine_search(
            tbl, qv, args.topk, "is_fulltext = true AND is_image = false"
        )
        elapsed = time.time() - t0
        times.append(elapsed)

        est = est_tokens(q) + sum(est_tokens(r.get("text", "")) for r in res[:3]) + 200
        tokens.append(est)

        # 语义命中率: 仅含 gt 的题统计, 空 gt 记 None 不拉低均值 (#1)
        rec = semantic_recall([r.get("text", "") for r in res], gt) if gt and res else None
        hits.append(rec)
        print(
            f"{idx+1:3d} {q[:34]:34s} {elapsed:6.3f}s "
            f"{('—' if rec is None else f'{rec:.3f}'):>6s}"
        )

        results.append(
            {
                "q": q,
                "time_s": round(elapsed, 4),
                "semantic_hit": rec,
                "est_tokens": est,
                "retrieved": [r.get("paper_id", "") for r in res[: args.topk]],
            }
        )

    hit_vals = [r for r in hits if r is not None]
    avg_hit = float(np.mean(hit_vals)) if hit_vals else 0.0
    gte80 = sum(1 for r in hit_vals if r >= 0.8)
    print(f'\n{"="*60}')
    print(f"总计: {total} 题 | 含标注: {len(hit_vals)} | KB: {rows} rows")
    print(
        f"检索延迟: avg={np.mean(times):.3f}s "
        f"p50={np.median(times):.3f}s max={max(times):.3f}s"
    )
    print(f"语义命中率(含标注): {avg_hit:.4f} (gte80: {gte80}/{len(hit_vals)})")
    print(
        f"预估 Token/题: {int(np.mean(tokens))} "
        f"| 预估 $/题: ${np.mean(tokens)/1e6*MODEL_COST_PER_1M:.5f}"
    )

    # 结构化 JSON 报告 (#9)
    try:
        report = {
            "kb": kb,
            "rows": rows,
            "qa_total": total,
            "qa_labeled": len(hit_vals),
            "latency": {
                "avg": round(float(np.mean(times)), 4),
                "p50": round(float(np.median(times)), 4),
                "max": round(float(max(times)), 4),
            },
            "semantic_hit": round(avg_hit, 4),
            "gte80": gte80,
            "est_tokens_avg": int(np.mean(tokens)),
            "est_cost_per_q": round(np.mean(tokens) / 1e6 * MODEL_COST_PER_1M, 6),
            "items": results,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {args.report}")
    except Exception as e:
        print(f"\n[警告] 报告保存失败: {e}")


asyncio.run(bench())
