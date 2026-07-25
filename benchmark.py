"""性能基准测试：响应时间 + 召回率 + Token 用量。"""
import sys, asyncio, time, json
sys.path.insert(0, '.')
from knowledge_base import _embed_texts_sync, _cosine_search, LANCEDB_DIR
import lancedb, numpy as np

async def bench():
    with open('data/eval_qa.json') as f: qa = json.load(f)
    db = lancedb.connect(str(LANCEDB_DIR))
    raw = db.list_tables()
    tables = raw.tables if hasattr(raw, 'tables') else []
    kb = [t for t in tables if 'eval' in str(t)]
    kb = kb[0] if kb else next((t for t in tables if 'papers' in str(t)), tables[0])
    tbl = db.open_table(kb)
    rows = tbl.count_rows()
    print(f'KB: {kb} ({rows} rows)\n')

    total = len(qa)
    times, recalls, tokens = [], [], []
    print(f'{"Question":40s} {"Time":>8s} {"Recall":>8s}')

    for item in qa:
        q, gt = item['question'], item.get('ground_truth', '')
        t0 = time.time()
        qv = _embed_texts_sync([q])[0]
        res = _cosine_search(tbl, qv, 5, "is_fulltext = true AND is_image = false")
        elapsed = time.time() - t0
        times.append(elapsed)

        est_tokens = len(q) // 4 + sum(len(r.get('text','')) // 4 for r in res[:3]) + 200
        tokens.append(est_tokens)

        rec = 0.0
        if gt and res:
            gt_emb = np.array(_embed_texts_sync([gt])[0])
            scores = []
            for r in res:
                te = np.array(_embed_texts_sync([r.get('text','')[:300]])[0])
                scores.append(float(np.dot(te, gt_emb) / (np.linalg.norm(te) * np.linalg.norm(gt_emb))))
            rec = max(scores) if scores else 0
        recalls.append(rec)
        print(f'{q[:40]:40s} {elapsed:7.3f}s {rec:7.4f}')

    print(f'\n{"="*60}')
    print(f'总计: {total} 题 | KB: {rows} rows')
    print(f'检索延迟: avg={np.mean(times):.3f}s  p50={np.median(times):.3f}s  max={max(times):.3f}s')
    print(f'Semantic Recall@5: {np.mean(recalls):.4f}  (gte80: {sum(1 for r in recalls if r>=0.8)}/{total})')
    print(f'预估 Token/题: {int(np.mean(tokens))}  | 预估 $/题: ${np.mean(tokens)/1e6*0.14:.5f}')

asyncio.run(bench())
