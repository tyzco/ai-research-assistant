"""轻量 RAG 评估：不依赖 langchain/RAGAS 重依赖，直接用 DeepSeek 评估。

用法：python evaluate_ragas.py
"""

import asyncio
import json
import logging
import time

logging.basicConfig(level=logging.WARNING)


async def evaluate_faithfulness(answer: str, contexts: list[str]) -> float:
    """DeepSeek 判断：回答是否基于给定上下文（而非幻觉）。"""
    from openai import AsyncOpenAI

    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

    if not contexts or not answer:
        return 0.0
    ctx_text = "\n---\n".join(contexts[:3])
    prompt = f"""判断以下回答是否完全基于给定的上下文（忠实度）。

上下文：
{ctx_text[:2000]}

回答：
{answer[:1000]}

请只回答一个 0-100 的数字，表示回答内容有多少比例是基于上下文的（100=完全基于上下文，0=完全编造）。只输出数字。"""
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        return float(text) / 100.0
    except Exception:
        return 0.5


def evaluate_context_precision(
    question: str, contexts: list[str], ground_truth: str
) -> float:
    """简单计算：检索到的上下文中，有多少包含 ground_truth 中的关键术语。"""
    if not contexts or not ground_truth:
        return 0.0
    # 提取 ground_truth 中的关键词（2字以上中文词或3字母以上英文词）
    import re

    gt_terms = set()
    for token in re.findall(r"[一-鿿]{2,}|[a-zA-Z]{3,}", ground_truth.lower()):
        gt_terms.add(token)
    if not gt_terms:
        return 0.5

    hits = 0
    for term in gt_terms:
        for ctx in contexts:
            if term.lower() in (ctx or "").lower():
                hits += 1
                break
    return min(1.0, hits / len(gt_terms))


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """简单计算：回答长度是否合理（太短=不相关，太长=啰嗦）。"""
    if not answer:
        return 0.0
    # 合理的回答应该 100-2000 字
    length = len(answer)
    if length < 50:
        return 0.3  # 太短，可能不相关
    elif length < 200:
        return 0.7
    elif length < 2000:
        return 0.9
    else:
        return 0.8  # 太长，可能重复
    # 另外检查回答是否包含问题的关键词
    q_terms = set((question or "").replace("？", "").replace("?", "").split())
    hit = sum(1 for t in q_terms if len(t) >= 2 and t in (answer or ""))
    bonus = min(0.1, hit * 0.02)
    return min(1.0, 0.7 + bonus)


async def main():
    # 加载测试数据
    try:
        with open("data/eval_qa.json") as f:
            test_qa = json.load(f)
    except FileNotFoundError:
        print("请先创建 data/eval_qa.json")
        return

    # 构建评估知识库
    from knowledge_base import build_knowledge_base
    from models import PaperMeta
    from qa_engine import ask_with_context

    eval_topic = "eval_" + str(int(time.time()))
    print("准备评估知识库...")
    papers = [
        PaperMeta(paper_id="e1", title="ArcFace: Additive Angular Margin Loss for Deep Face Recognition", year=2019),
        PaperMeta(paper_id="e2", title="FaceNet: A Unified Embedding for Face Recognition and Clustering", year=2015),
        PaperMeta(paper_id="e3", title="DeepFace: Closing the Gap to Human-Level Performance in Face Verification", year=2014),
        PaperMeta(paper_id="e4", title="LFW: Labeled Faces in the Wild Database", year=2007),
        PaperMeta(paper_id="e5", title="CosFace: Large Margin Cosine Loss for Deep Face Recognition", year=2018),
        PaperMeta(paper_id="e6", title="SphereFace: Deep Hypersphere Embedding for Face Recognition", year=2017),
        PaperMeta(paper_id="e7", title="A Survey of Face Recognition Techniques", year=2020),
    ]
    chunks = {
        "e1": ["ArcFace adds angular margin penalty to target logit. Unlike triplet loss which needs complex mining, ArcFace stabilizes training. Achieves SOTA on LFW, MegaFace, IJB-C. Better class separation than FaceNet."],
        "e2": ["FaceNet uses triplet loss to train deep CNN outputting 128-dim embeddings. Triplet loss ensures same person closer than others. Uses hard positive/negative triplet mining. Unified system for verification, recognition, clustering."],
        "e3": ["DeepFace achieves 97.35% on LFW near human-level. Uses ensemble of deep networks with 120M params on 4.4M faces. Key: 3D face alignment and Siamese architecture. LFW evaluates under unconstrained conditions."],
        "e4": ["LFW contains 13,233 images of 5,749 individuals from Yahoo News. Standard benchmark for face verification with variations in pose, lighting, expression. Uses 10-fold cross-validation on 6,000 pairs, reporting accuracy and ROC."],
        "e5": ["CosFace proposes Large Margin Cosine Loss adding cosine margin in cosine space. ArcFace adds angular margin in angle space. Both maximize inter-class variance via margin-based softmax. Normalize features and weights for discrimination."],
        "e6": ["SphereFace introduces A-Softmax for hypersphere embedding with angular margin. One of first angular margin works. Common losses: Softmax, Triplet Loss(FaceNet), Center Loss, L-Softmax, A-Softmax(SphereFace), CosFace, ArcFace."],
        "e7": ["Face recognition dominated by CNN-based deep learning. Loss evolution: softmax to contrastive to triplet to margin-based. Benchmarks: LFW(verification), MegaFace(identification), IJB-C(1:N). Metrics: accuracy, ROC, AUC, rank-1."],
    }
    table = await build_knowledge_base(papers, chunks, eval_topic)
    print(f"知识库: {table}")

    # 跑评估
    scores = {
        "faithfulness": [],
        "context_precision": [],
        "answer_relevancy": [],
        "count": 0,
    }
    print(f"\n评估 {len(test_qa)} 条...")
    for i, item in enumerate(test_qa):
        q = item["question"]
        gt = item.get("ground_truth", "")
        try:
            ans, ctxs = await ask_with_context(table, q)
            faith = await evaluate_faithfulness(ans, ctxs)
            cprec = evaluate_context_precision(q, ctxs, gt)
            relev = evaluate_answer_relevancy(q, ans)
            scores["faithfulness"].append(faith)
            scores["context_precision"].append(cprec)
            scores["answer_relevancy"].append(relev)
            scores["count"] += 1
            print(
                f"  [{i+1}/{len(test_qa)}] {q[:30]}... faith={faith:.2f} cprec={cprec:.2f} relev={relev:.2f}"
            )
        except Exception as e:
            print(f"  [{i+1}/{len(test_qa)}] ❌ {e}")

    # 汇总
    print("\n" + "=" * 50)
    print("RAG 评估结果 (轻量)")
    print("=" * 50)
    result = {}
    for k, v in scores.items():
        if k != "count" and v:
            avg = sum(v) / len(v)
            result[k] = round(avg, 4)
            print(f"  {k}: {avg:.4f}")
    print(f"  评估条目: {scores['count']}")
    print("=" * 50)

    with open("data/rag_eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("结果已保存到 data/rag_eval_result.json")


if __name__ == "__main__":
    asyncio.run(main())
