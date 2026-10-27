"""在**合成检索基准**上评测: 客观命中判定 + 置信区间 + 消融。

与 run_eval.py 的区别:
  run_eval    人工 8 题, ground truth 是"语义相似度近似" —— 只能当回归门禁
  本脚本      每题有唯一 gold_chunk_id —— 命中与否是**客观事实**, 且 N 可做大,
              配合 bootstrap 置信区间与消融表, 结论才立得住

指标:
  Recall@5/10   gold chunk 是否出现在前 k
  MRR           gold chunk 首次出现的排名倒数
  NDCG@10       二值相关性下 1/log2(rank+1)

消融 (同一批题、同一 KB):
  vector       纯向量
  hybrid       自适应 RRF (BM25 + 向量)
  hybrid+r1    上面 + Stage1 (MiniLM 预筛)
  hybrid+r2    上面 + 两级级联 (即完整链路)

用法:
    python run_retrieval_bench.py --bench data/retrieval_bench.json --kb papers_benchmark_eval_kb
    python run_retrieval_bench.py --bench ... --limit 20      # 冒烟
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")

F = "is_fulltext = true AND is_image = false"


def _recall_at(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def _ndcg_at(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _bootstrap_ci(vals: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """百分位 bootstrap 95% CI。N 小时区间会明显变宽 —— 这本身就是要展示的信息。"""
    if not vals:
        return (0.0, 0.0)
    rnd = random.Random(42)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        means.append(sum(vals[rnd.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[int((1 - alpha / 2) * len(means)) - 1]
    return (lo, hi)


def _rank_of(rows: list[dict], gold: str) -> int | None:
    for i, r in enumerate(rows, 1):
        if (r.get("chunk_id") or "") == gold:
            return i
    return None


def evaluate(hr, items: list[dict], mode: str) -> dict:
    """mode: vector | hybrid | hybrid_no_stage1 | hybrid_full"""
    rec5, rec10, mrr, ndcg = [], [], [], []
    for it in items:
        q, gold = it["question"], it["gold_chunk_id"]
        try:
            if mode == "vector":
                rows = hr._vector_recall(q, 10, F)
            elif mode == "hybrid":
                # 关掉级联重排, 只看融合结果
                saved = hr._cascade_rerank
                hr._cascade_rerank = lambda query, chunks, top_k=5, stage1_k=None: chunks[:top_k]
                try:
                    rows, _ = hr.search_sync(q, top_k=10, filter_expr=F)
                finally:
                    hr._cascade_rerank = saved
            elif mode == "hybrid_no_stage2":
                # 只做 Stage1 预筛, 跳 Stage2 精排
                hr._cascade_rerank = _stage1_only(hr)
                try:
                    rows, _ = hr.search_sync(q, top_k=10, filter_expr=F)
                finally:
                    del hr._cascade_rerank
            else:  # hybrid_full
                rows, _ = hr.search_sync(q, top_k=10, filter_expr=F)
        except Exception:
            rows = []
        r = _rank_of(rows or [], gold)
        rec5.append(_recall_at(r, 5))
        rec10.append(_recall_at(r, 10))
        mrr.append(1.0 / r if r else 0.0)
        ndcg.append(_ndcg_at(r, 10))
    out = {"n": len(items)}
    for name, vals in (("recall@5", rec5), ("recall@10", rec10), ("mrr", mrr), ("ndcg@10", ndcg)):
        lo, hi = _bootstrap_ci(vals)
        out[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
        out[name + "_ci"] = [round(lo, 4), round(hi, 4)]
    return out


def _stage1_only(hr):
    """返回一个只做 Stage1 的 _cascade_rerank 替身 (用于消融)。"""

    def _fn(query, chunks, top_k=5, stage1_k=None):
        if len(chunks) <= top_k:
            return chunks
        keep = min(stage1_k or max(top_k * 5, 25), len(chunks))
        try:
            import numpy as np

            m = hr._get_stage1()
            q_emb = m.encode([query])[0]
            c_embs = m.encode([c.get("text", "")[:512] for c in chunks])
            qn = q_emb / (np.linalg.norm(q_emb) + 1e-12)
            cn = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-12)
            scores = np.dot(cn, qn)
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
            return [chunks[i] for i in idx][:top_k]
        except Exception:
            return chunks[:top_k]

    return _fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/retrieval_bench.json")
    ap.add_argument("--kb", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    p = Path(args.bench)
    if not p.exists():
        print(f"❌ 基准不存在: {p}（先跑 build_retrieval_benchmark.py）")
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("❌ 基准为空")
        return 1
    kb = args.kb or data.get("kb", "")

    from hybrid_retriever import AdaptiveHybridRetriever

    hr = AdaptiveHybridRetriever(kb)
    print(f"KB={kb} | 题数={len(items)} | 协议={data.get('protocol','')}")
    print(f"{'配置':<18} {'R@5':>7} {'R@5 95%CI':>16} {'R@10':>7} {'MRR':>7} {'NDCG@10':>8}")
    modes = [
        ("vector", "纯向量"),
        ("hybrid", "自适应 RRF"),
        ("hybrid_no_stage2", "+Stage1 预筛"),
        ("hybrid_full", "+两级级联(完整)"),
    ]
    report = {"kb": kb, "n": len(items), "protocol": data.get("protocol", ""), "results": {}}
    for mode, label in modes:
        r = evaluate(hr, items, mode)
        report["results"][mode] = r
        ci = r["recall@5_ci"]
        print(
            f"{label:<16} {r['recall@5']:>7.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] "
            f"{r['recall@10']:>7.4f} {r['mrr']:>7.4f} {r['ndcg@10']:>8.4f}"
        )

    out = Path("data/retrieval_bench_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {out}")
    print("提示: 合成基准偏乐观, 报告时需与人工集 data/eval_qa.json 分开列。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
