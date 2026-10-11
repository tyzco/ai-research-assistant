"""人工验证样本导出: 打破"自测自证"闭环, 加外部人工锚点。

做法:
  1. 跑 N 题 → 每题输出 问题/答案/检索上下文(节选)/judge 逐论断分类
  2. 存成可标注文件 (data/review_samples.csv + .md 预览)
  3. 人工标注两列: human_answer_ok (答案是否正确/切题: 1/0), human_grounded (论断是否真被上下文支撑: 1/0/NA)
  4. 标注后用 compute_review_metrics.py 算人工验证指标 + judge-人工一致性

用法:
    python export_review_samples.py --kb papers_big_eval --n 10
    # 填 CSV 的 human_* 列后:
    python export_review_samples.py --metrics
"""
import argparse
import asyncio
import csv
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from eval_common import load_eval_qa  # noqa: E402
from claim_diagnose import _split_claims, _judge_support, _retrieve_kb  # noqa: E402

REVIEW_FILE = "data/review_samples.csv"
REVIEW_JSON = "data/review_samples.json"


def _summarize_ctx(contexts: list[dict], k: int = 3) -> str:
    """检索上下文节选 (给人类判断接地性用)。"""
    parts = []
    for c in contexts[:k]:
        t = c.get("title", "")
        txt = (c.get("text", "") or "")[:250].replace("\n", " ")
        parts.append(f"[{t}] {txt}")
    return "\n".join(parts)


async def _collect_samples(qa: list[dict], table: str, limit: int) -> list[dict]:
    from qa_engine import ask_question, _rerank
    from hybrid_retriever import AdaptiveHybridRetriever
    from knowledge_base import retrieve_with_images

    hr = AdaptiveHybridRetriever(table)
    samples = []
    items = qa[:limit] if limit else qa
    for i, item in enumerate(items, 1):
        q = item.get("question", "")
        try:
            res = (await ask_question(table, q))
            answer = res.answer
            # 检索上下文
            tr, _, _ = retrieve_with_images(table, q, top_k=15)
            try:
                tr = await asyncio.to_thread(_rerank, q, tr, 8)
            except Exception:
                pass
            ctx_rows = tr[:8]
            # judge 逐论断分类
            ctx_texts = [c.get("text", "")[:900] for c in ctx_rows]
            claims = _split_claims(answer)
            claim_judge = []
            for cl in claims:
                rel = _judge_support(cl, ctx_texts)
                claim_judge.append({"claim": cl[:150], "judge": rel})
            samples.append({
                "id": i,
                "question": q[:200],
                "answer": answer[:1800],
                "contexts": _summarize_ctx(ctx_rows),
                "n_claims": len(claims),
                "claim_judge": claim_judge,
                "human_answer_ok": "",
                "human_grounded": "",
                "human_notes": "",
            })
            print(f"[{i}/{len(items)}] {q[:30]} → {len(claims)} 论断 (judge supported率见 claims)")
        except Exception as e:
            print(f"[{i}] ERR {e}")
    return samples


def _save(samples: list[dict]):
    with open(REVIEW_JSON, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)
    with open(REVIEW_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "contexts", "human_answer_ok(1/0)", "human_grounded(1/0)", "human_notes"])
        for s in samples:
            w.writerow([s["id"], s["question"], s["answer"], s["contexts"],
                        s["human_answer_ok"], s["human_grounded"], s["human_notes"]])
    print(f"\n已导出 {len(samples)} 题 → {REVIEW_FILE} + {REVIEW_JSON}")
    print("人工标注列: human_answer_ok(答案是否正确切题:1/0), human_grounded(答案论断是否真被上下文支撑:1/0)")
    print("填完后跑: python export_review_samples.py --metrics")


def _compute_metrics():
    """读人工标注, 算: 人工答案正确率 / 人工接地率 / judge-人工一致性。"""
    samples = json.load(open(REVIEW_JSON, encoding="utf-8"))
    # 尝试从 CSV 读人工标注合并
    try:
        rows = list(csv.DictReader(open(REVIEW_FILE, encoding="utf-8-sig")))
        by_id = {int(r["id"]): r for r in rows}
        for s in samples:
            r = by_id.get(s["id"], {})
            s["human_answer_ok"] = r.get("human_answer_ok(1/0)", "").strip()
            s["human_grounded"] = r.get("human_grounded(1/0)", "").strip()
            s["human_notes"] = r.get("human_notes", "").strip()
    except Exception as e:
        print(f"读 CSV 失败: {e}")

    graded = [s for s in samples if s.get("human_answer_ok") in ("1", "0")]
    if not graded:
        print("尚未标注 human_answer_ok。请先填写 review_samples.csv 的 human_answer_ok 列。")
        return

    ans_ok = sum(1 for s in graded if s["human_answer_ok"] == "1")
    print(f"\n===== 人工验证指标 ({len(graded)} 题已标) =====")
    print(f"答案正确率 (human_answer_ok=1): {ans_ok}/{len(graded)} = {ans_ok/len(graded):.2f}")

    # judge-人工一致性 (对每题 judge 的 supported率 vs 人工接地)
    grounded = [s for s in samples if s.get("human_grounded") in ("1", "0")]
    if grounded:
        agree = 0
        for s in grounded:
            judge_supported = sum(1 for c in s.get("claim_judge", []) if c["judge"] == "supported")
            judge_ok = judge_supported >= max(1, s.get("n_claims", 1) // 2)  # judge: 多数论断支撑=接地
            human_ok = s["human_grounded"] == "1"
            if judge_ok == human_ok:
                agree += 1
        print(f"接地性 judge-人工一致性: {agree}/{len(grounded)} = {agree/len(grounded):.2f}")
    print("\n注: 一致性是粗略代理 (judge=多数论断支撑), 用于展示评估非自证。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="papers_big_eval")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--metrics", action="store_true", help="算人工验证指标")
    ap.add_argument(
        "--labeled",
        action="store_true",
        help="导出**人工标注过的** bad case 作为回归集 (用户反馈 → 知识库迭代的回流出口)",
    )
    args = ap.parse_args()
    if args.labeled:
        return export_labeled_regression()

    if args.metrics:
        _compute_metrics()
        return
    qa = load_eval_qa("data/benchmark_qa.json", 0)
    print(f"导出 {args.n} 题人工验证样本 (KB={args.kb}, 每题约25s, 需几分钟)...")
    samples = asyncio.run(_collect_samples(qa, args.kb, args.n))
    if samples:
        _save(samples)


def export_labeled_regression(out_path: str = "data/labeled_regression.json") -> int:
    """把**人工标注过的** bad case 导出成回归集 —— 用户反馈闭环的出口。

    为什么需要: 在此之前, 标注只到 `/review/stats` 的分布统计为止, **没有任何东西消费它**
    —— 反馈回来的一手失败样本既没进评测集, 也没影响切片/检索策略, 闭环是断的。
    导出后可用它跑回归: 同一批真实失败问题在改动前后各跑一次, 看是否真的修好了。

    导出内容包括当时检索到的上下文 (review_queue.contexts), 便于对比改动前后的召回差异。
    只导出标注过的 (status='labeled'), pending 的不算 —— 未经人看的样本不能当基准。
    """
    from database import get_db

    items = get_db().list_review_queue(status="labeled", limit=500)
    if not items:
        print("暂无已标注样本 (先在 /review/queue 标注后再导出)")
        return 0

    rows = []
    for it in items:
        rows.append(
            {
                "question": it.get("question", ""),
                "label": it.get("label", ""),
                "note": it.get("label_note", ""),
                "source": it.get("source", ""),
                "reason": it.get("reason", ""),
                # 标注当时的检索上下文 —— 供改动后对比召回是否变化
                "contexts_at_label_time": it.get("contexts", []),
                "labeled_at": it.get("labeled_at", ""),
            }
        )
    import json as _json
    from collections import Counter as _C

    from pathlib import Path as _P

    _P(out_path).parent.mkdir(parents=True, exist_ok=True)
    _P(out_path).write_text(
        _json.dumps({"n": len(rows), "items": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dist = _C(r["label"] for r in rows)
    print(f"已导出 {len(rows)} 条人工标注样本 → {out_path}")
    print("标签分布: " + ", ".join(f"{k}={v}" for k, v in dist.most_common()))
    if dist.get("retrieval_miss"):
        print(
            f"  注: {dist['retrieval_miss']} 条 retrieval_miss (库里有但没召回) —— "
            "这类是切片/检索策略最直接的改进信号"
        )
    return len(rows)


if __name__ == "__main__":
    main()
