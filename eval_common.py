"""评估脚本共享工具 — run_eval / benchmark / evaluate_rag_standards 复用。

消除三份重复实现 (数据加载 / KB 选择 / 语义召回 / MRR)。
用法:
    from eval_common import load_eval_qa, pick_kb, semantic_recall, mrr
"""
import json
import re

import numpy as np

import lancedb

from config import LANCEDB_DIR
from knowledge_base import _embed_texts_sync


def load_eval_qa(path: str = "data/eval_qa.json", limit: int = 0) -> list[dict]:
    """加载并校验测试集; 跳过缺 question 的条目, 支持 --limit。"""
    try:
        with open(path, encoding="utf-8") as f:
            qa = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"测试集不存在: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"测试集 JSON 解析失败: {e}")
    valid = []
    for i, item in enumerate(qa):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            print(f"[跳过] 第 {i} 条缺少 question")
            continue
        valid.append(item)
    if limit > 0:
        valid = valid[:limit]
    if not valid:
        raise SystemExit("测试集无有效条目")
    return valid


def list_tables() -> list[str]:
    """LanceDB 表列举 (兼容不同版本 API)。"""
    db = lancedb.connect(str(LANCEDB_DIR))
    raw = db.list_tables()
    return list(raw.tables) if hasattr(raw, "tables") else [str(t) for t in raw]


# 临时知识库标记: 这些由评测/冒烟脚本创建, 内容极少 (常常只有 1 篇论文),
# 一旦被 pick_kb 选中, 任何问题都能"召回"到唯一那篇 → R@5=1.0 / MRR=1.0 的假指标。
# 实测踩过: 一次被 kill 的评测留下 papers_agent_eval_xxx, 之后 run_eval 就一直在
# 那份单篇库上跑, 报告显示 MRR=1.0, 而真实值约 0.67。
_TEMP_KB_MARKERS = ("agent_eval_", "e2e_", "probe_", "smoke_")


def _is_temp_kb(name: str) -> bool:
    return any(m in str(name) for m in _TEMP_KB_MARKERS)


def pick_kb(cli_kb: str = "", prefer: tuple = ("eval", "papers")) -> str:
    """选择评估表: CLI 指定 > prefer 子串 > 首个。**排除临时库**。"""
    tables = [t for t in list_tables() if not _is_temp_kb(t)]
    if cli_kb:
        if cli_kb not in tables:
            raise SystemExit(f"指定的 KB 不存在: {cli_kb}")
        return cli_kb
    if not tables:
        raise SystemExit("LanceDB 无任何表, 请先构建知识库")
    for pref in prefer:
        m = [t for t in tables if pref in str(t)]
        if m:
            return m[0]
    return tables[0]


def _embed_one(text: str) -> np.ndarray:
    return np.array(_embed_texts_sync([text])[0])


def semantic_recall(contexts: list[str], ground_truth: str, top_k: int = 5) -> float:
    """检索上下文与 ground_truth 的最大语义相似度 (top-k 内)。

    注: 这是"语义相似度"近似, 非标准 Recall@k (需标注相关文档)。
    保留此命名以与项目内其他脚本一致, 但在报告中标明其近似性。
    """
    if not contexts or not ground_truth:
        return 0.0
    ge = _embed_one(ground_truth)
    ce = np.array(_embed_texts_sync([c[:500] for c in contexts[:top_k]]))
    sims = np.dot(ce, ge) / (np.linalg.norm(ce, axis=1) * np.linalg.norm(ge) + 1e-12)
    return float(np.max(sims))


def mrr(contexts: list[str], ground_truth: str, threshold: float = 0.75) -> float:
    """第一个相似度 >= threshold 的排名的倒数 (MRR 近似)。"""
    if not contexts or not ground_truth:
        return 0.0
    ge = _embed_one(ground_truth)
    for i, c in enumerate(contexts[:10]):
        ce = _embed_one(c[:500])
        s = float(np.dot(ce, ge) / (np.linalg.norm(ce) * np.linalg.norm(ge) + 1e-12))
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(contexts: list[str], ground_truth: str, top_k: int = 5, threshold: float = 0.75) -> float:
    """NDCG@k — 带位置折扣的排序指标 (IR 标准), 二值相关度。

    注: 不用连续相似度作 gain — bge 相似度挤在 ~0.7-0.9 窄带, 连续 gain 会让 NDCG 恒≈1
    (排序几乎不影响, 无信息量)。改用二值相关度 gain_i = 1(相似度≥阈值) else 0:
    衡量"相关块是否排在前面", 与 MRR 同阈值可比, 但对全部相关块的位置都加权(MRR 只看第一个)。
    """
    if not contexts or not ground_truth:
        return 0.0
    ge = _embed_one(ground_truth)
    gains = []
    for c in contexts[:top_k]:
        ce = _embed_one((c or "")[:500])
        s = float(np.dot(ce, ge) / (np.linalg.norm(ce) * np.linalg.norm(ge) + 1e-12))
        gains.append(1.0 if s >= threshold else 0.0)
    if not gains or sum(gains) == 0:
        return 0.0
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 1e-12 else 0.0


def est_tokens(text: str) -> int:
    """CN/EN 感知 token 估算: 中文 1.5 char/token, 其他 3.5。"""
    if not text:
        return 0
    cn = len(re.findall(r"[一-鿿]", text))
    return int(cn / 1.5 + (len(text) - cn) / 3.5)
