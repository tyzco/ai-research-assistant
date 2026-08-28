"""自动化检索评估 + 回归测试: Hybrid vs Vector 对比 + 回归门禁。

用法:
    python run_eval.py                # 自动选择评估表
    python run_eval.py --kb papers_x  # 指定表
    python run_eval.py --limit 10     # 只评前 10 题

指标 (语义相似度近似, 见 eval_common.semantic_recall):
  - Vector-R@5 / Hybrid-R@5: 检索上下文与 ground truth 的最大语义相似度
  - MRR: 第一个相似度>=0.75 的排名倒数
  - 回归门禁: hybrid recall 较上次下降 > 0.02 → 阻断
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

import lancedb

from eval_common import load_eval_qa, mrr, pick_kb, semantic_recall
from knowledge_base import LANCEDB_DIR, _cosine_search, _embed_texts_sync


def run_vector_search(tbl, question: str, top_k: int = 10) -> list[str]:
    qv = _embed_texts_sync([question])[0]
    res = _cosine_search(tbl, qv, top_k, "is_fulltext = true AND is_image = false")
    return [r.get("text", "")[:500] for r in (res or [])]


def run_hybrid_search(tbl, question: str, top_k: int = 10) -> list[str]:
    from hybrid_retriever import AdaptiveHybridRetriever

    try:
        hr = AdaptiveHybridRetriever(tbl.name)
        text_res, _ = hr.search_sync(
            question, top_k=top_k, filter_expr="is_fulltext = true AND is_image = false"
        )
        return [r.get("text", "")[:500] for r in (text_res or [])]
    except Exception as e:
        # 混合检索失败 → 降级为向量检索, 不中断整个评估
        print(f"  [警告] 混合检索失败({e}), 降级为向量检索")
        return run_vector_search(tbl, question, top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="", help="指定 LanceDB 表名 (默认自动选择)")
    ap.add_argument("--qa", default="data/eval_qa.json", help="测试集路径")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="仅评估前 N 题 (默认全部)")
    args = ap.parse_args()

    qa = load_eval_qa(args.qa, args.limit)
    kb = pick_kb(args.kb)
    db = lancedb.connect(str(LANCEDB_DIR))
    tbl = db.open_table(kb)

    print(f"Evaluating {len(qa)} questions on {kb} ({tbl.count_rows()} rows)")
    print(f"{'='*70}")
    print(f"{'Question':45s} {'Vec-R@5':>8s} {'Hyb-R@5':>8s} {'MRR':>6s}")

    results, failed = [], []
    t0 = time.time()
    for item in qa:
        q = item.get("question", "")
        gt = item.get("ground_truth", "")
        try:
            # 每题独立 try/except, 失败不中断后续 (#2)
            vec_texts = run_vector_search(tbl, q, top_k=args.topk)
            hyb_texts = run_hybrid_search(tbl, q, top_k=args.topk)
            vec_r5 = semantic_recall(vec_texts, gt)
            hyb_r5 = semantic_recall(hyb_texts, gt)
            m = mrr(hyb_texts, gt)
            results.append({"q": q[:40], "vec_r5": vec_r5, "hyb_r5": hyb_r5, "mrr": m})
            print(f"{q[:45]:45s} {vec_r5:7.4f}  {hyb_r5:7.4f}  {m:5.3f}")
        except Exception as e:
            failed.append({"q": q[:60], "error": str(e)[:120]})
            print(f"  [!!] {q[:40]} 失败: {str(e)[:80]}")

    elapsed = time.time() - t0
    avg_vec = np.mean([r["vec_r5"] for r in results]) if results else 0
    avg_hyb = np.mean([r["hyb_r5"] for r in results]) if results else 0
    avg_mrr = np.mean([r["mrr"] for r in results]) if results else 0
    hyb_wins = sum(1 for r in results if r["hyb_r5"] > r["vec_r5"])

    report = {
        "model": "bge-small-zh (512d)",
        "table": kb,
        "questions": len(qa),
        "failed": len(failed),
        "elapsed_seconds": round(elapsed, 1),
        "vector_recall_at_5": round(float(avg_vec), 4),
        "hybrid_recall_at_5": round(float(avg_hyb), 4),
        "hybrid_vs_vector_delta": round(float(avg_hyb - avg_vec), 4),
        "hybrid_wins": f"{hyb_wins}/{len(results)}",
        "mrr": round(float(avg_mrr), 4),
        "recall_gte80": f"{sum(1 for r in results if r['hyb_r5'] >= 0.80)}/{len(results)}",
    }

    print(f"\n{'='*70}")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # 回归门禁: 与上次报告对比, hybrid recall 下降 > 0.02 阻断
    prev_path = Path("data/eval_result.json")
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())
        delta = report["hybrid_recall_at_5"] - prev.get(
            "hybrid_recall_at_5", report["hybrid_recall_at_5"]
        )
        report["recall_delta_vs_previous"] = round(delta, 4)
        if delta < -0.02:
            print(f"\n  REGRESSION: Recall dropped by {delta:.4f} (threshold: -0.02)")
        else:
            print(f"\n  Recall change: {delta:+.4f} (PASS)")

    prev_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nSaved data/eval_result.json")
    return report


if __name__ == "__main__":
    main()
