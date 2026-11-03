"""RAG 标准评估 (RAGAS 指标, 自研实现): faithfulness + answer_relevancy + context_precision + context_recall。

用法:
    python evaluate_rag_standards.py                    # benchmark_qa 全部 30 题
    python evaluate_rag_standards.py --limit 8          # 只评前 8 题 (快)
    python evaluate_rag_standards.py --kb papers_xxx    # 指定 KB (默认选内容最多的非 eval 表)

为什么不用 ragas 框架 (已试过, 记录在此):
  - RAGAS 的 LLM judge 绑定 OpenAI 输出格式, 与 DeepSeek 不兼容 → faithfulness 恒 0 /
    answer_relevancy nan / context_recall 卡死。
  - 故指标保留, 判定全部自研 (与项目"自研 RAG"一致):
      faithfulness / answer_relevancy → DeepSeek 温度0 直判 (thinking disabled, 确定性)
      context_recall → 嵌入语义召回 (近似)
      context_precision → top-k 命中 ground_truth 相关上下文占比 (嵌入阈值)
  - 每个样本会走真实管线 (ask_with_context: 检索+重排+生成答案) 再打分, 较慢 (约 20-30s/题)。
"""
import argparse
from pathlib import Path
import asyncio
import json
import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("rag_standards")
logging.basicConfig(level=logging.INFO)

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_QUICK_MAX_TOKENS,
)
from eval_common import load_eval_qa, pick_kb, semantic_recall  # noqa: E402
from knowledge_base import _embed_texts_sync  # noqa: E402


# ============ DeepSeek 直判评分器 (确定性: 温度0 + thinking disabled) ============

def _judge_llm(prompt: str, max_tokens: int = 60, timeout: int = 30) -> str:
    """DeepSeek 直判, 返回原始输出。失败返回 ""。"""
    from openai import OpenAI

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.0,
        max_tokens=min(LLM_QUICK_MAX_TOKENS, max_tokens),
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


def _parse_score(raw: str) -> float | None:
    """从 LLM 输出里提取 0-1 分数 (鲁棒: 数字/分数/JSON 均可)。"""
    import re

    if not raw:
        return None
    m = re.search(r"\d+\.?\d*", raw)
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


def judge_faithfulness(answer: str, contexts: list[str]) -> float | None:
    """答案论断被上下文支撑的比例 (0-1)。RAGAS faithfulness 语义。

    按论断逐个判定并取被支撑比例 (部分支撑给部分分), 避免"有任何不支持即 0"的二值化误杀。
    """
    ctx = "\n".join(f"- {c[:900]}" for c in contexts[:8]) or "（无上下文）"
    prompt = (
        "你是 RAG 评估器。将答案拆成若干论断, 逐个判断每个论断是否被给定上下文"
        "直接支撑或可由上下文合理推断。\n"
        "规则: 上下文完全没有依据的论断 (编造/外推) 算不被支撑; 上下文为空则该答案所有论断都不被支撑。\n"
        f"上下文:\n{ctx}\n\n答案:\n{(answer or '')[:1800]}\n\n"
        "只输出被支撑比例 = 被支撑论断数 / 总论断数, 一个 0-1 之间的小数, 不要解释。"
    )
    return _parse_score(_judge_llm(prompt))


def judge_answer_relevancy(question: str, answer: str) -> float | None:
    """答案是否直接回应了问题 (0-1)。RAGAS answer_relevancy 语义。"""
    prompt = (
        "你是 RAG 评估器。判断答案是否直接、完整地回答了问题。\n"
        "规则: 答非所问/跑题/回避 → 低分; 完整覆盖问题核心 → 高分。\n"
        f"问题: {question[:300]}\n\n答案: {(answer or '')[:1500]}\n\n"
        "只输出一个 0-1 之间的数 (0=完全无关, 1=完整回答), 不要解释。"
    )
    return _parse_score(_judge_llm(prompt))


def context_precision(contexts: list[str], ground_truth: str, top_k: int = 5, thresh: float = 0.75) -> float:
    """top-k 中与 ground_truth 相关的上下文占比 (嵌入语义近似 RAGAS context_precision)。"""
    if not contexts or not ground_truth:
        return 0.0
    ge = _embed_texts_sync([ground_truth])[0]
    top = contexts[:top_k]
    ce = np.array(_embed_texts_sync([c[:500] for c in top]))
    sims = np.dot(ce, ge) / (np.linalg.norm(ce, axis=1) * np.linalg.norm(ge) + 1e-12)
    return float(np.mean(sims >= thresh))


# ============ 主评估流程 ============

async def run_eval(qa_list: list[dict], table_name: str, limit: int = 0) -> dict:
    from qa_engine import ask_with_context

    items = qa_list[:limit] if limit else qa_list
    rows = []
    for i, item in enumerate(items, 1):
        q = item.get("question", "")
        gt = item.get("ground_truth", "")
        try:
            answer, contexts = await ask_with_context(table_name, q)
        except Exception as e:
            logger.warning(f"QA failed [{i}]: {q[:30]}: {e}")
            rows.append({"question": q, "error": str(e)})
            continue
        f = judge_faithfulness(answer, contexts)
        r = judge_answer_relevancy(q, answer)
        cr = semantic_recall(contexts, gt, top_k=5)
        cp = context_precision(contexts, gt, top_k=5)
        rows.append(
            {
                "question": q,
                "answer_len": len(answer or ""),
                "contexts": len(contexts),
                "faithfulness": f,
                "answer_relevancy": r,
                "context_recall": cr,
                "context_precision": cp,
            }
        )
        logger.info(
            f"[{i}/{len(items)}] faith={f if f is not None else 'NA':.2f} "
            f"rel={r if r is not None else 'NA':.2f} rec={cr:.3f} prec={cp:.3f}"
        )

    def _avg(key: str, filter_none: bool = True) -> float:
        vals = [x.get(key) for x in rows if x.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    report = {
        "kb": table_name,
        "n": len(rows),
        "questions": len(items),
        "faithfulness": _avg("faithfulness"),
        "answer_relevancy": _avg("answer_relevancy"),
        "context_recall": _avg("context_recall"),
        "context_precision": _avg("context_precision"),
        "note": "faithfulness/relevancy=DeepSeek直判; context_recall/precision=嵌入语义近似 (非标准标注).",
    }
    with open("data/rag_standards_report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": report, "rows": rows}, f, ensure_ascii=False, indent=1)
    return report


async def eval_overconfidence(qa_list: list[dict], table_name: str) -> dict:
    """量"依据不足时是否硬答" —— 这批题目的答案在知识库里**不存在**。

    为什么单列一项: RAGAS 四指标衡量的是"有据可依时答得好不好", 但**依据不足时是否
    敢说不知道**是另一个维度。实测过最危险的失败形态: 内层判定"库内无相关内容"、
    外层却用参数化知识写出看似严谨的答案 —— 整体分数可能很高, 但那是幻觉伪装成引用。
    所以这里只看两件事: 有没有如实说明资料不足; 有没有在无依据时给出结论性回答。
    """
    from qa_engine import ask_with_context

    MARK = "【库内无相关内容】"
    rows = []
    for item in qa_list:
        q = item.get("question", "")
        try:
            answer, _ = await ask_with_context(table_name, q)
        except Exception as e:
            rows.append({"question": q, "error": str(e)})
            continue
        refused = MARK in (answer or "")
        rows.append(
            {
                "question": q,
                "refused": refused,
                "overconfident": not refused,
                "answer_len": len(answer or ""),
                "answer_preview": (answer or "")[:160],
            }
        )
    valid = [r for r in rows if "error" not in r]
    n = len(valid) or 1
    return {
        "n": len(valid),
        "refusal_rate": round(sum(1 for r in valid if r["refused"]) / n, 4),
        # 越高越危险: 库内明明没有, 却给出了结论性回答
        "overconfidence_rate": round(sum(1 for r in valid if r["overconfident"]) / n, 4),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kb", type=str, default="")
    ap.add_argument("--qa", type=str, default="benchmarks/eval_qa_30.json")
    ap.add_argument("--topic", type=str, default="", help="只评该主题(可逗号分隔, 如 '强化学习,大语言模型')")
    ap.add_argument(
        "--unanswerable",
        type=str,
        default="",
        metavar="QA_JSON",
        help="只跑**库外问题**切片, 报告 refusal_rate / overconfidence_rate "
        "(如 data/eval_unanswerable.json)",
    )
    args = ap.parse_args()

    if args.unanswerable:
        from eval_common import load_eval_qa as _load

        uq = _load(args.unanswerable, limit=args.limit or 0)
        table = args.kb or pick_kb(prefer=("benchmark", "eval", "papers"))
        print(f"库外问题切片: {len(uq)} 题 | KB: {table}")
        rep = asyncio.run(eval_overconfidence(uq, table))
        print("\n===== 库外问题 (依据不足) 评估 =====")
        print(f"  如实说明资料不足: {rep['refusal_rate']:.0%}")
        print(f"  依据不足仍硬答  : {rep['overconfidence_rate']:.0%}  ← 越低越好")
        for r in rep["rows"]:
            flag = "❌ 硬答" if r.get("overconfident") else "✅ 如实说明"
            print(f"    {flag} {r['question'][:32]}")
        import json as _json

        _p = Path("data/overconfidence_report.json")
        _p.write_text(_json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存 {_p}")
        return

    qa = load_eval_qa(args.qa, limit=0)
    if args.topic:
        topics = {t.strip() for t in args.topic.split(",") if t.strip()}
        qa = [x for x in qa if x.get("topic", "") in topics]
    table = args.kb or pick_kb(prefer=("benchmark", "eval", "papers"))
    print(f"评估 KB: {table}, 题目: {len(qa)} (limit={args.limit or '全部'}, topic={args.topic or '全部'})")
    report = asyncio.run(run_eval(qa, table, args.limit))
    print("\n===== RAG 标准评估汇总 =====")
    for k in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
        print(f"  {k:20s}: {report.get(k)}")
    print(f"已保存 data/rag_standards_report.json (n={report['n']})")


if __name__ == "__main__":
    main()
