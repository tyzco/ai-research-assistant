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


def pick_kb(cli_kb: str = "", prefer: tuple = ("eval", "papers")) -> str:
    """选择评估表: CLI 指定 > prefer 前缀 > 首个。空表优雅退出。"""
    tables = list_tables()
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


def est_tokens(text: str) -> int:
    """CN/EN 感知 token 估算: 中文 1.5 char/token, 其他 3.5。"""
    if not text:
        return 0
    cn = len(re.findall(r"[一-鿿]", text))
    return int(cn / 1.5 + (len(text) - cn) / 3.5)
