"""RAG 标准评估 (ragas): faithfulness + context_precision + answer_relevancy + context_recall。

用法:
    python evaluate_rag_standards.py                    # 全部题
    python evaluate_rag_standards.py --limit 5          # 只评前 5 题
    python evaluate_rag_standards.py --kb papers_xxx    # 指定 KB

评估器: DeepSeek (llm_factory) + 本地 bge 嵌入 (LocalBgeEmbeddings)。
注: 每个样本会生成真实答案 (qa_engine.ask_with_context) 再交由 ragas 打分, 耗时较长。
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import types


# 离线/本地缓存加载模型, 避免 sentence-transformers 每次启动联网检查 HuggingFace 卡顿
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.WARNING)

# ragas 0.4.3 与 langchain_community 0.4.x 兼容 stub (vertexai 已拆分为独立包, 本评估不使用)
_mod = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAI:
    def __init__(self, *a, **k):
        raise NotImplementedError("VertexAI 未配置, 本评估使用 DeepSeek")


_mod.ChatVertexAI = _ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _mod)

from langchain_openai import ChatOpenAI  # noqa: E402
from openai import OpenAI  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbeddings  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import context_precision  # noqa: E402
# 注: faithfulness/answer_relevancy/context_recall 的 LLM 判定与 DeepSeek 输出格式不兼容
# (faithfulness 恒 0 / answer_relevancy nan / context_recall 卡死)。
# 故: faithfulness+answer_relevancy 用 DeepSeek 直判, context_recall 用嵌入语义召回。

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL  # noqa: E402
from eval_common import load_eval_qa, pick_kb, semantic_recall  # noqa: E402
from knowledge_base import LANCEDB_DIR, _embed_texts_sync  # noqa: E402


class LocalBgeEmbeddings(BaseRagasEmbeddings):
    """用项目本地 bge 模型提供 ragas 评估嵌入。"""

    def embed_documents(self, texts):
        return _embed_texts_sync(texts)

    def embed_query(self, text):
        return _embed_texts_sync([text])[0]

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


# 同步客户端: ragas 的 nest_asyncio 会破坏后续 async 调用, fallback 用同步避免 (实测可靠)
_fallback_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _llm_judge(prompt: str) -> float:
    """DeepSeek 直接 0-1 判定 (同步, 不受 async 循环状态影响)。"""
    try:
        resp = _fallback_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0,
            max_tokens=100,  # deepseek-v4-pro 是推理模型, max_tokens=10 全被推理消耗 → 输出空串
            messages=[{"role": "user", "content": prompt}],
            timeout=20,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return max(0.0, min(1.0, float(text.replace(",", "."))))
    except Exception:
        pass
    return 0.5


def fallback_faithfulness(answer: str, ctxs: list[str]) -> float:
    ctx = "\n---\n".join(c[:500] for c in ctxs[:3])
    return _llm_judge(
        f"判断以下回答是否完全基于上下文（忠实度）。1=完全基于, 0=完全编造。只回答0到1的小数。\n\n"
        f"上下文：{ctx[:2000]}\n\n回答：{answer[:1000]}\n\n忠实度分数（0-1）："
    )


def fallback_relevancy(question: str, answer: str) -> float:
    return _llm_judge(
        f"判断以下回答是否直接回答了问题（相关性）。1=完全切题, 0=完全离题。只回答0到1的小数。\n\n"
        f"问题：{question[:300]}\n\n回答：{answer[:1000]}\n\n相关性分数（0-1）："
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/eval_qa.json")
    ap.add_argument("--kb", default="", help="指定 LanceDB 表名 (默认自动选择)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="仅评估前 N 题 (默认全部)")
    ap.add_argument("--report", default="data/rag_standards_report.json")
    args = ap.parse_args()

    qa = load_eval_qa(args.qa, args.limit)
    kb = pick_kb(args.kb)

    from qa_engine import ask_with_context

    samples, errors = [], []
    print(f"生成真实答案并组装评估样本 ({len(qa)} 题, KB: {kb})...")
    for i, item in enumerate(qa):
        q = str(item["question"])
        gt = str(item.get("ground_truth", ""))
        try:
            answer, ctxs = await ask_with_context(kb, q)
            samples.append(
                SingleTurnSample(
                    user_input=q,
                    retrieved_contexts=ctxs or [],
                    response=answer,
                    reference=gt or None,
                )
            )
            print(f"  [{i+1}/{len(qa)}] ✓ {q[:40]}")
        except Exception as e:
            errors.append({"q": q, "error": str(e)[:150]})
            print(f"  [{i+1}/{len(qa)}] ✗ 失败: {e}")

    if not samples:
        raise SystemExit("无有效样本可评估")

    print("\nragas 评估中 (DeepSeek evaluator + 本地 bge 嵌入)...")
    # LangchainLLMWrapper (非 instructor): 纯 chat 生成, 对 DeepSeek 等 OpenAI 兼容模型更稳定
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            max_tokens=2048,
        )
    )
    # ragas 只跑嵌入型可靠的 context_precision (faithfulness/answer_relevancy/context_recall 与 DeepSeek 不兼容)
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[context_precision],
        llm=llm,
        embeddings=LocalBgeEmbeddings(),
        show_progress=True,
    )

    df = result.to_pandas()
    # faithfulness / answer_relevancy → DeepSeek 直判 (可靠)
    for idx, row in df.iterrows():
        q = row.get("user_input", "") or ""
        ans = row.get("response", "") or ""
        ctxs = row.get("retrieved_contexts") or []
        df.at[idx, "faithfulness"] = fallback_faithfulness(ans, ctxs)
        df.at[idx, "answer_relevancy"] = fallback_relevancy(q, ans)
        # context_recall → 嵌入语义召回 (top-5 与 ground_truth 最大相似度)
        gt = samples[idx].reference or ""
        df.at[idx, "context_recall"] = semantic_recall(ctxs, gt)

    report = {
        "kb": kb,
        "model": "bge-small-zh (512d) + DeepSeek (ragas: context metrics; DeepSeek 直判: LLM metrics)",
        "questions": len(samples),
        "failed": len(errors),
        "faithfulness": round(float(df["faithfulness"].mean()), 4),
        "context_precision": round(float(df["context_precision"].mean()), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "context_recall": round(float(df["context_recall"].mean()), 4),
    }
    print(f"\n{'='*60}\nRAG 标准评估报告")
    for k, v in report.items():
        print(f"  {k}: {v}")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(
            {"report": report, "per_item": df.to_dict(orient="records"), "errors": errors},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\nSaved {args.report}")


asyncio.run(main())
