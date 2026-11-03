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
import asyncio
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


Q_TIMEOUT = 60  # 单题超时: 没有它, 一个卡住的调用会让整批判测静默挂死


def make_retriever(kb: str, mode: str):
    """按消融模式构造检索器, **在并发开始前**一次性打补丁。

    不能在每题内部临时替换 `_cascade_rerank`: 并发下多个协程会互相覆盖对方的补丁,
    消融结果就串了 (早期版本正是每查询 patch 一次)。
    """
    from hybrid_retriever import AdaptiveHybridRetriever

    hr = AdaptiveHybridRetriever(kb)
    if mode == "hybrid":
        # 关掉级联重排, 只看融合结果
        hr._cascade_rerank = (
            lambda query, chunks, top_k=5, stage1_k=None: chunks[:top_k]
        )
    elif mode == "hybrid_no_stage2":
        # 只做 Stage1 预筛, 跳 Stage2 精排
        hr._cascade_rerank = _stage1_only(hr)
    return hr


async def _run_query(hr, q: str, mode: str) -> list[dict]:
    if mode == "vector":
        # 纯向量: 不涉及 LLM, 放线程池避免阻塞事件循环
        return await asyncio.to_thread(hr._vector_recall, q, 10, F)
    rows, _ = await asyncio.wait_for(
        hr.search(q, top_k=10, filter_expr=F), timeout=Q_TIMEOUT
    )
    return rows or []


async def evaluate(hr, items: list[dict], mode: str, concurrency: int = 6) -> dict:
    """并发评测; 每题独立计时, 单题失败记为未命中并继续。

    并发只发生在**本地 CPU 检索**上 (评测已关闭外部翻译, 见 QUERY_TRANSLATION_ENABLED),
    因此不会重演"共享 LLM 客户端连接池被跨 loop 拖垮"的挂死。
    """
    sem = asyncio.Semaphore(concurrency)
    metrics: list[tuple[float, float, float, float]] = []
    done = [0]

    async def _one(it: dict) -> None:
        async with sem:
            try:
                rows = await _run_query(hr, it["question"], mode)
            except Exception:
                rows = []
            r = _rank_of(rows, it["gold_chunk_id"])
            metrics.append(
                (
                    _recall_at(r, 5),
                    _recall_at(r, 10),
                    1.0 / r if r else 0.0,
                    _ndcg_at(r, 10),
                )
            )
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"    {mode}: {done[0]}/{len(items)}", flush=True)

    await asyncio.gather(*[_one(it) for it in items])
    rec5 = [m[0] for m in metrics]
    rec10 = [m[1] for m in metrics]
    mrr = [m[2] for m in metrics]
    ndcg = [m[3] for m in metrics]
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
    ap.add_argument("--bench", default="benchmarks/retrieval_bench_800.json")
    ap.add_argument("--kb", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
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

    print(f"KB={kb} | 题数={len(items)} | 并发={args.concurrency} | 协议={data.get('protocol','')}")
    print(f"{'配置':<18} {'R@5':>7} {'R@5 95%CI':>16} {'R@10':>7} {'MRR':>7} {'NDCG@10':>8}")
    modes = [
        ("vector", "纯向量"),
        ("hybrid", "自适应 RRF"),
        ("hybrid_no_stage2", "+Stage1 预筛"),
        ("hybrid_full", "+两级级联(完整)"),
    ]
    report = {"kb": kb, "n": len(items), "protocol": data.get("protocol", ""), "results": {}}
    async def _run_all():
        for mode, label in modes:
            hr = make_retriever(kb, mode)
            r = await evaluate(hr, items, mode, args.concurrency)
            report["results"][mode] = r
            ci = r["recall@5_ci"]
            print(
                f"{label:<16} {r['recall@5']:>7.4f} "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] "
                f"{r['recall@10']:>7.4f} {r['mrr']:>7.4f} {r['ndcg@10']:>8.4f}",
                flush=True,
            )

    asyncio.run(_run_all())

    out = Path("data/retrieval_bench_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {out}")
    print("提示: 合成基准偏乐观, 报告时需与人工集 data/eval_qa.json 分开列。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
