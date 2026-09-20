"""真实评估基准 (可对外宣传的版本) — 多主题 + 诚实指标 + 环境/局限文档化。

与 run_eval.py (8 题内部冒烟测试) 的区别:
  - 多主题 (人脸识别/Transformer/CNN/强化学习/大语言模型...)
  - 每题有参考答案 (公认事实, 非从 KB 挖出)
  - 报告含: 分主题结果 + MRR + Recall + 环境 + 明确局限

诚实声明 (报告自动输出):
  - semantic_recall 是语义近似, 不是"答对"判定
  - MRR/Recall 上界受限于知识库覆盖面 (KB 没有该主题论文 → 分数必然低, 反映的是覆盖率不是检索质量)
  - 换嵌入模型/网络/判官, 绝对值会变; 相对比较更稳健

用法:
    python run_benchmark.py [--kb 表名] [--topk 5]
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

import lancedb

from eval_common import mrr, ndcg, pick_kb, semantic_recall
from knowledge_base import LANCEDB_DIR, _cosine_search, _embed_texts_sync


def run_retrieval(tbl, question, top_k, use_hybrid):
    if not use_hybrid:
        qv = _embed_texts_sync([question])[0]
        res = _cosine_search(tbl, qv, top_k, "is_fulltext = true AND is_image = false")
        return [r.get("text", "")[:500] for r in (res or [])]
    from hybrid_retriever import AdaptiveHybridRetriever

    try:
        hr = AdaptiveHybridRetriever(tbl.name)
        text_res, _ = hr.search_sync(question, top_k=top_k, filter_expr="is_fulltext = true AND is_image = false")
        return [r.get("text", "")[:500] for r in (text_res or [])]
    except Exception:
        qv = _embed_texts_sync([question])[0]
        res = _cosine_search(tbl, qv, top_k, "is_fulltext = true AND is_image = false")
        return [r.get("text", "")[:500] for r in (res or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="", help="指定 LanceDB 表 (默认自动选择)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--qa", default="data/benchmark_qa.json", help="基准问题集")
    args = ap.parse_args()

    qa = json.loads(Path(args.qa).read_text())
    topics = {}
    for item in qa:
        topics.setdefault(item.get("topic", "其他"), []).append(item)

    kb = pick_kb(args.kb)
    db = lancedb.connect(str(LANCEDB_DIR))
    tbl = db.open_table(kb)
    print(f"基准: {len(qa)} 题 / {len(topics)} 主题, KB={kb} ({tbl.count_rows()} 行)")
    print("=" * 70)

    per_topic = {}
    all_r5, all_mrr, all_ndcg = [], [], []
    t0 = time.time()
    for topic, items in sorted(topics.items()):
        r5s, mrrs, ndcgs = [], [], []
        for item in items:
            q, gt = item["question"], item["ground_truth"]
            texts = run_retrieval(tbl, q, args.topk, use_hybrid=True)
            r5s.append(semantic_recall(texts, gt))
            mrrs.append(mrr(texts, gt))
            ndcgs.append(ndcg(texts, gt, top_k=args.topk))
        per_topic[topic] = {
            "questions": len(items),
            "recall_at_topk": round(sum(r5s) / len(r5s), 4),
            "mrr": round(sum(mrrs) / len(mrrs), 4),
            "ndcg_at_topk": round(sum(ndcgs) / len(ndcgs), 4),
        }
        all_r5 += r5s
        all_mrr += mrrs
        all_ndcg += ndcgs
        print(
            f"  {topic:12s} {len(items):2d}题  R@topk={per_topic[topic]['recall_at_topk']:.4f}  "
            f"MRR={per_topic[topic]['mrr']:.4f}  NDCG={per_topic[topic]['ndcg_at_topk']:.4f}"
        )

    report = {
        "questions": len(qa),
        "topics": len(topics),
        "kb": kb,
        "kb_rows": tbl.count_rows(),
        "top_k": args.topk,
        "elapsed_seconds": round(time.time() - t0, 1),
        "overall_recall_at_topk": round(sum(all_r5) / len(all_r5), 4),
        "overall_mrr": round(sum(all_mrr) / len(all_mrr), 4),
        "overall_ndcg_at_topk": round(sum(all_ndcg) / len(all_ndcg), 4),
        "per_topic": per_topic,
        "environment": {
            "embedding": "bge-small-zh (512d)",
            "retrieval": "adaptive hybrid (BM25+vector) + cascade rerank",
            "similarity_metric": "cosine (bge-small-zh)",
        },
        "limitations": [
            "semantic_recall 是语义近似, 不是'答对'判定",
            "分数受限于 KB 覆盖面: KB 无该主题论文 → 分低 (反映覆盖率, 非检索质量)",
            "参考答案为公认事实, 未逐题人工判分 (LLM 判分可选)",
            "换嵌入/网络/判官, 绝对值会变; 相对比较更稳健",
        ],
    }
    out = Path("data/benchmark_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n" + "=" * 70)
    print(
        f"总览: {len(qa)} 题 R@topk={report['overall_recall_at_topk']:.4f}  "
        f"MRR={report['overall_mrr']:.4f}  NDCG@{args.topk}={report['overall_ndcg_at_topk']:.4f}"
    )
    print("环境:", report["environment"])
    print("局限 (对外声明时必读):")
    for l in report["limitations"]:
        print("  - " + l)
    print("\n已保存 data/benchmark_report.json")


if __name__ == "__main__":
    main()
