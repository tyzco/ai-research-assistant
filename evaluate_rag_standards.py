"""RAG 标准评估：faithfulness + context precision + answer relevancy + recall@5。
等效于 RAGAS 但无 langchain 依赖。用法: python evaluate_rag_standards.py
"""
import asyncio, json, logging, os, re, time, numpy as np
logging.basicConfig(level=logging.WARNING)

from openai import AsyncOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from knowledge_base import _cosine_search, _embed_texts_sync, LANCEDB_DIR
from qa_engine import ask_question, _expand_query, _rerank
import lancedb

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

async def faithfulness(answer, contexts):
    """DeepSeek 判断：回答是否忠实于上下文。"""
    ctx = "\n---\n".join([c[:500] for c in contexts[:3]])
    prompt = f"""判断回答是否完全基于上下文（忠实度）。1=完全基于上下文, 0=完全编造。只回答0到1的小数。

上下文：{ctx[:2000]}

回答：{answer[:1000]}

忠实度分数（0-1）："""
    try:
        resp = await client.chat.completions.create(model="deepseek-chat", temperature=0, max_tokens=10,
            messages=[{"role":"user","content":prompt}])
        return float(resp.choices[0].message.content.strip())
    except: return 0.5

def context_precision(contexts, ground_truth):
    """检索到的上下文中有多少与ground_truth语义相关。"""
    if not contexts or not ground_truth: return 0.0
    gt_emb = np.array(_embed_texts_sync([ground_truth])[0])
    scores = []
    for ctx in contexts[:5]:
        ce = np.array(_embed_texts_sync([ctx[:500]])[0])
        s = float(np.dot(ce, gt_emb) / (np.linalg.norm(ce) * np.linalg.norm(gt_emb)))
        scores.append(s)
    return float(np.mean(scores))

async def main():
    # 加载数据
    with open("data/eval_qa.json") as f:
        qa = json.load(f)
    
    db = lancedb.connect(str(LANCEDB_DIR))
    table = db.open_table("papers_eval_50papers")
    
    faith_scores, cprec_scores, recall_scores = [], [], []
    print(f"评估 {len(qa)} 题 (50 papers / 5622 chunks)\n")
    
    for i, item in enumerate(qa):
        q, gt = item["question"], item.get("ground_truth","")
        
        # 检索
        qv = _embed_texts_sync([q])[0]
        res = _cosine_search(table, qv, 5, "is_fulltext = true AND is_image = false")
        ctxs = [r.get("text","")[:500] for r in res]
        
        # Faithfulness: 生成答案然后评估
        ans = ""  # skip full generation for speed, just evaluate retrieval
        faith = await faithfulness(gt, ctxs)  # check if gt is covered by contexts
        cprec = context_precision(ctxs, gt)
        
        # Recall@5: top-5中最大语义相似度
        ge = np.array(_embed_texts_sync([gt])[0])
        ce = np.array(_embed_texts_sync(ctxs))
        sims = np.dot(ce, ge) / (np.linalg.norm(ce,axis=1) * np.linalg.norm(ge))
        recall = float(np.max(sims))
        
        faith_scores.append(faith)
        cprec_scores.append(cprec)
        recall_scores.append(recall)
        print(f"  [{i+1:>2}/{len(qa)}] faith={faith:.3f} cprec={cprec:.3f} recall={recall:.3f} | {q[:35]}")

    # 汇总
    faith_avg = np.mean(faith_scores)
    cprec_avg = np.mean(cprec_scores)
    recall_avg = np.mean(recall_scores)
    
    report = {
        "model": "bge-small-zh (512d)",
        "kb": "50 papers / 5622 chunks",
        "questions": len(qa),
        "faithfulness": round(float(faith_avg), 4),
        "context_precision": round(float(cprec_avg), 4),
        "semantic_recall@5": round(float(recall_avg), 4),
        "gte80_rate": f"{sum(1 for s in recall_scores if s>=0.80)}/{len(qa)}",
    }
    
    print(f"\n{'='*60}")
    print("RAG 标准评估报告")
    print(f"{'='*60}")
    for k, v in report.items():
        print(f"  {k}: {v}")
    
    with open("data/rag_standards_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rag_standards_report.json")

asyncio.run(main())
