"""构建**可核查**的检索基准: 每个问题有唯一的 gold_chunk_id。

为什么不用现有 8 题人工集:
  8 题的绝对指标两头不讨好 —— 低了被看不上, 高了被质疑评测是否科学。根因是
  (a) N 太小, 没有统计意义; (b) ground truth 是"语义相似度近似", 不是客观答案。

本脚本的做法 (标准合成检索基准):
  1. 从知识库抽样 N 个全文块
  2. 让模型**只根据这一块**生成一个具体问题
  3. gold = 该块的 chunk_id —— 客观、可自动判定, 不存在"判官偏见"
  4. 记录生成所用的块, 便于人工抽查

诚实声明 (要写进报告):
  - 合成问题, 不是真实用户查询; 措辞会贴近原文, 指标**偏乐观**
  - 因此同时保留人工集 (data/eval_qa.json) 单独报告
  - 生成时要求模型改写而非照抄, 但仍不能等同真实分布

用法:
    python build_retrieval_benchmark.py --kb papers_benchmark_eval_kb --n 100
    python build_retrieval_benchmark.py --kb ... --n 20 --concurrency 4   # 冒烟
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")

BENCH_PATH = Path("data/retrieval_bench.json")

PROMPT = """下面是一段学术论文原文。请根据它提出**一个**具体问题。

要求：
1. 问题必须能且只能由这段原文回答（不要问这段里没有的内容）
2. 问题要具体（含术语/对象），不要问"这段讲了什么"
3. **改写**成自然提问，不要照抄原文的整句；不要出现"这段/上述/本文/根据材料"等指代
4. 只输出问题本身，一行，不要解释、不要编号

原文：
{chunk}
"""


def _rows(kb: str) -> list[dict]:
    import lancedb

    from config import LANCEDB_DIR
    from knowledge_base import _get_table

    db = lancedb.connect(str(LANCEDB_DIR))
    t = _get_table(db, kb)
    rows = t.search().where("is_fulltext = true AND is_image = false").limit(100000).to_list()
    # 太短的块信息量不足, 生成的问题会很空
    return [r for r in rows if len((r.get("text") or "").strip()) >= 300]


async def _gen_one(client, sem, row: dict, model: str, timeout: int) -> dict | None:
    async with sem:
        chunk = (row.get("text") or "").strip()[:3000]
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT.format(chunk=chunk)}],
                    temperature=0.2,
                    max_tokens=120,
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                timeout=timeout,
            )
            q = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  生成失败: {type(e).__name__}: {str(e)[:60]}", file=sys.stderr)
            return None
    q = q.split("\n")[0].strip().strip("「」\"'")
    # 过滤: 太短 / 带指代 / 明显没生成问题
    if len(q) < 8 or len(q) > 120:
        return None
    if re.search(r"这段|上述|本文|根据材料|以上内容", q):
        return None
    return {
        "question": q,
        "gold_chunk_id": row.get("chunk_id") or "",
        "source_title": (row.get("title") or "")[:80],
        "source_section": (row.get("section") or "")[:60],
        # 留一小段原文供人工抽查"问题是否真的只能由它回答"
        "source_preview": chunk[:200],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--n", type=int, default=100, help="目标题数")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(BENCH_PATH))
    args = ap.parse_args()

    from config import DEEPSEEK_MODEL
    from llm_client import get_client

    rows = _rows(args.kb)
    if not rows:
        print(f"❌ {args.kb} 没有可用的全文块")
        return 1
    print(f"KB={args.kb} 可用块 {len(rows)}，目标生成 {args.n} 题")

    # 多取一些候选, 生成失败的用后面的补
    random.Random(args.seed).shuffle(rows)
    cand = rows[: min(len(rows), int(args.n * 1.3) + 20)]

    client = get_client()
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    results = await asyncio.gather(
        *[_gen_one(client, sem, r, DEEPSEEK_MODEL, args.timeout) for r in cand]
    )
    items = [x for x in results if x][: args.n]

    if not items:
        print("❌ 一条都没生成成功")
        return 1

    # 去重: 同 gold chunk 只保留一条; 问题文本重复也去掉
    seen_chunk, seen_q, uniq = set(), set(), []
    for x in items:
        k = x["gold_chunk_id"]
        qn = re.sub(r"\s+", "", x["question"])
        if k in seen_chunk or qn in seen_q:
            continue
        seen_chunk.add(k)
        seen_q.add(qn)
        uniq.append(x)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "kb": args.kb,
                "protocol": "synthetic: question generated from the source chunk; gold = chunk_id",
                "disclosure": "合成问题措辞贴近原文, 指标偏乐观; 需与人工集分开报告",
                "n": len(uniq),
                "items": uniq,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ 生成 {len(uniq)} 题（去重后），用时 {time.time()-t0:.0f}s → {out}")
    print("   抽样:")
    for x in uniq[:5]:
        print(f"   - {x['question'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
