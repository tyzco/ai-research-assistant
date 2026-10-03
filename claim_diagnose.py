"""论断级错误诊断: 把每个答案论断对「检索上下文 vs 全库」两级判定, 分解三类失败杠杆。

思路 (deep-research 结论落地): faithfulness 低是生成侧结构问题, 但要修哪里需要量化:
  每个论断 →
    (1) 检索上下文 (top-5) 是否支撑?  → supported ✅
    (2) 不支撑的论断, 用论断本身在全库新检索 top-3 是否支撑?
         支撑 → 🔍 in-KB-not-retrieved  (检索问题: 信息在库里但原检索没接住)
         不支撑 → 🧠 not-in-KB  (生成侧: 模型参数知识, 库里没有)
    (3) 上下文与论断矛盾 → ⚠️ contradicted  (生成侧知识冲突)

输出: 各分类占比 → 决定先修检索 / 修生成 / 接受。

用法:
    python claim_diagnose.py --kb papers_big_eval [--limit 6] [--topic X]
"""
import argparse
import asyncio
import json
import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

from eval_common import load_eval_qa  # noqa: E402
from evaluate_rag_standards import _judge_llm, _parse_score  # noqa: E402


def _split_claims(answer: str) -> list[str]:
    """用 DeepSeek 把答案拆成原子论断 (每个可独立判定支撑性)。"""
    prompt = (
        "将以下答案拆成若干个原子论断(claim), 每个论断是一个可单独判定'是否被证据支撑'的陈述句。\n"
        "规则: 不合并、不概括、保留原文关键信息; 一条论断一句。\n"
        "输出: 每行一个论断, 不要编号, 不要解释。\n\n答案:\n" + (answer or "")[:2000]
    )
    raw = _judge_llm(prompt, max_tokens=400, timeout=40)
    claims = [l.strip("- •0123456789. ").strip() for l in (raw or "").splitlines()]
    return [c for c in claims if len(c) >= 4][:8]


def _judge_support(claim: str, contexts: list[str]) -> str:
    """判定论断相对上下文的支撑关系: supported / contradicted / unaddressed。"""
    ctx = "\n".join(f"- {c[:350]}" for c in contexts[:4]) or "（无上下文）"
    prompt = (
        "判断论断与给定证据的关系。三类:\n"
        "- supported: 论断被证据直接支撑或可由证据合理推断\n"
        "- contradicted: 论断与证据明确矛盾\n"
        "- unaddressed: 证据未提及该论断 (证据既不支撑也不反驳)\n"
        f"证据:\n{ctx}\n\n论断: {claim}\n\n只输出一个词 (supported/contradicted/unaddressed)。"
    )
    raw = (_judge_llm(prompt, max_tokens=10, timeout=30) or "").strip().lower()
    for k in ("contradicted", "unaddressed", "supported"):
        if k in raw:
            return k
    return "unaddressed"


def _retrieve_kb(table_name: str, claim: str, k: int = 3) -> list[dict]:
    """用论断本身检索全库, 判断信息是否在库里 (检索层没接住 vs 库外知识)。

    用 `search_sync`: 本函数是同步的, 而 `search` 是 `async def` —— 调它只会拿到一个
    coroutine 对象, 解包即 TypeError, 又被下面的 except 吞掉, 导致本函数**永远返回 []**
    (grounding 因此永不触发)。旁边本就有同步版 `search_sync`, 之前用错了。
    """
    from hybrid_retriever import AdaptiveHybridRetriever

    hr = AdaptiveHybridRetriever(table_name)
    try:
        res, _ = hr.search_sync(claim, top_k=k)
        return res
    except Exception as e:
        logger.warning(f"_retrieve_kb failed: {e}")
        return []


async def run_diagnose(qa: list[dict], table_name: str, limit: int = 0) -> dict:
    from qa_engine import ask_with_context

    items = qa[:limit] if limit else qa
    counts = {"supported": 0, "in_kb_not_retrieved": 0, "not_in_kb": 0, "contradicted": 0}
    total_claims = 0
    samples = []

    for idx, item in enumerate(items, 1):
        q = item.get("question", "")
        try:
            answer, ctxs = await ask_with_context(table_name, q)
        except Exception as e:
            print(f"[{idx}] QA err: {e}")
            continue
        claims = _split_claims(answer)
        total_claims += len(claims)
        print(f"[{idx}/{len(items)}] {q[:30]} → {len(claims)} 论断")
        for c in claims:
            rel = _judge_support(c, ctxs)
            cls = rel
            if rel == "supported":
                cls = "supported"
            elif rel == "contradicted":
                cls = "contradicted"
            else:  # unaddressed → 查全库
                kb_rows = _retrieve_kb(table_name, c)
                kb_ctx = [r.get("text", "") for r in kb_rows]
                rel2 = _judge_support(c, kb_ctx)
                cls = "in_kb_not_retrieved" if rel2 == "supported" else "not_in_kb"
            counts[cls] = counts.get(cls, 0) + 1
            print(f"    [{cls}] {c[:48]}")
            samples.append({"question": q[:60], "claim": c[:120], "class": cls})

    total = max(sum(counts.values()), 1)
    report = {
        "kb": table_name,
        "n_questions": len(items),
        "total_claims": total_claims,
        "split": {k: round(v / total, 3) for k, v in counts.items()},
        "counts": counts,
        "note": (
            "supported=检索上下文直接支撑; in_kb_not_retrieved=信息在库但原检索没接住(检索杠杆); "
            "not_in_kb=库外模型知识(生成侧, 需决策是否约束); contradicted=与检索上下文矛盾(生成侧知识冲突)"
        ),
    }
    with open("data/claim_diagnose_report.json", "w", encoding="utf-8") as f:
        json.dump({"report": report, "samples": samples}, f, ensure_ascii=False, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=str, default="papers_big_eval")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--qa", type=str, default="data/benchmark_qa.json")
    args = ap.parse_args()

    qa = load_eval_qa(args.qa, limit=0)
    print(f"论断级诊断: KB={args.kb}, 题数={min(args.limit or len(qa), len(qa))}")
    report = asyncio.run(run_diagnose(qa, args.kb, args.limit))
    print("\n===== 论断分类占比 =====")
    for k, v in report["split"].items():
        print(f"  {k:22s}: {v*100:.1f}%  ({report['counts'].get(k,0)})")
    print(f"已保存 data/claim_diagnose_report.json")


if __name__ == "__main__":
    main()
