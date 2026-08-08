"""第二层意图识别: 嵌入语义分类器 (三层漏斗的中间层)。

三层漏斗设计:
  1. 规则层 (毫秒级): config 关键词匹配
     - qa_engine._classify_intent / agent.skills.select_for_task
     - 快路径, 命中即返回, 不触发任何模型加载
  2. 语义层 (本模块): bge 嵌入 + 意图原型向量余弦相似度
     - 规则未命中时兜底, 承接"意思常规但改了表达"的查询
     - 低于阈值 → 返回 None, 交由第 3 层
  3. 大模型+工具层: pi_agent Function Calling 工具循环
     - 对复杂/模糊请求, 由 LLM 自主决定调用哪个工具

与规则层互补: 原型句式刻意选"关键词表里没有、但意图很常规"的问法,
例如 "这个模型的改进点主要有哪些" (不含 对比/比较/区别 等词, 语义上却是对比)。

设计约束:
  - 延迟加载: 首次语义分类才编码原型, 不影响 api.py 启动与纯规则快路径
  - 线程安全: 原型编码单次初始化 (双检锁, 同 embedder 模式)
  - 永不抛错: 集成点在 try/except 内; 语义层不可用时静默降级为 规则+LLM 两层
  - 归一化: 复用 embedder.embed_texts (已 L2 归一化), 相似度即点积

自测: python intent_classifier.py
"""

import logging
import threading
import unicodedata

logger = logging.getLogger(__name__)

# ===== 意图原型: 每类 5 条典型问法 =====
# 这些是"关键词容易漏、但意思很常规"的句式, 与 config.INTENT_KEYWORDS 互补。
# 新增意图只需在此加原型, 无需改集成点代码。

QA_INTENT_PROTOTYPES = {
    "定义": [
        "Transformer架构的核心组件有哪些",
        "什么是注意力机制",
        "这个概念的精确定义是什么",
        "扩散模型的数学形式",
        "LoRA到底指什么",
    ],
    "对比": [
        "这两种优化器哪个收敛更快",
        "对比不同注意力机制变体",
        "这个模型相对上个版本的改进点在哪",
        "RAG和微调哪种方案更适合",
        "多头注意力与单头的优劣",
    ],
    "实操": [
        "我该怎样复现这篇论文的实验",
        "用BERT做文本分类的完整流程",
        "怎么搭建模型训练环境",
        "这篇代码的调用步骤",
        "如何把模型部署成服务",
    ],
    "最新": [
        "2025年这个方向有哪些新突破",
        "最近半年有什么重要进展",
        "这个领域目前的前沿工作",
        "最新发表的综述覆盖了哪些内容",
        "今年有哪些值得关注的新模型",
    ],
    "原理": [
        "Transformer为什么能并行处理序列",
        "梯度消失背后的机制是什么",
        "这个损失函数设计的动机",
        "为什么微调会导致灾难性遗忘",
        "位置编码起作用的原理",
    ],
}

SKILL_PROTOTYPES = {
    "literature_review": [
        "帮我做一份大模型安全的文献综述",
        "调研图神经网络的研究现状",
        "搜索这个方向的论文并整理成报告",
        "当前该领域有哪些研究热点",
        "对这个方向做全面的文献调研",
    ],
    "paper_writing": [
        "帮我把实验结果写成论文",
        "生成一篇会议论文的大纲",
        "润色这段引言部分",
        "撰写论文的方法章节",
        "帮我整理参考文献格式",
    ],
    "deep_qa": [
        "解释知识蒸馏的原理",
        "LoRA微调是怎么工作的",
        "RAG和微调有什么区别",
        "大模型为什么会产生幻觉",
        "多头注意力机制详解",
    ],
    "research_decision": [
        "分析这个方向近五年的发表趋势",
        "哪些论文的引用数最高",
        "评估这篇工作的学术影响力",
        "我该不该投入这个研究方向",
        "对比几个候选方向的科研价值",
    ],
    "citation_exploration": [
        "这篇论文引用了哪些关键工作",
        "追踪Transformer的发展脉络",
        "谁引用了这篇奠基论文",
        "理清这个方向的引用关系链",
        "找这个方向的源头工作",
    ],
    "paper_critique": [
        "批判性分析这篇论文",
        "评估这篇工作的实验是否公平",
        "这篇论文的方法有什么弱点",
        "审阅这篇论文的主张和证据",
        "这篇工作值得相信吗",
    ],
    "experiment_design": [
        "为这个方向设计实验方案",
        "怎么设计消融实验",
        "应该用哪些基线和指标",
        "帮我规划实验和评测",
        "这个假设怎么验证",
    ],
}

QA_LABELS = tuple(QA_INTENT_PROTOTYPES.keys())
SKILL_LABELS = tuple(SKILL_PROTOTYPES.keys())


class _ProtoCache:
    """意图原型向量缓存 (延迟加载 + 双检锁, 同 embedder 模式)。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._label_vectors: dict[str, list[list[float]]] | None = None

    def vectors(self) -> dict[str, list[list[float]]]:
        """返回 label → 原型向量列表。嵌入不可用 → 返回空 dict (缓存失败, 不重试)。"""
        if self._label_vectors is not None:
            return self._label_vectors
        with self._lock:
            if self._label_vectors is not None:
                return self._label_vectors
            try:
                from embedder import embed_texts
            except Exception as e:
                logger.warning(f"Semantic intent layer unavailable (embedder): {e}")
                self._label_vectors = {}
                return self._label_vectors
            merged: dict[str, list[str]] = {}
            for src in (QA_INTENT_PROTOTYPES, SKILL_PROTOTYPES):
                for label, exs in src.items():
                    merged.setdefault(label, []).extend(exs)
            try:
                self._label_vectors = {
                    label: embed_texts(exs) for label, exs in merged.items()
                }
                logger.info(
                    f"Semantic intent prototypes loaded: "
                    f"{len(self._label_vectors)} labels "
                    f"({sum(len(v) for v in self._label_vectors.values())} examples)"
                )
            except Exception as e:
                logger.warning(f"Semantic intent prototypes failed to encode: {e}")
                self._label_vectors = {}
        return self._label_vectors


_proto_cache = _ProtoCache()

# 轻量统计 (监控用; GIL 下计数足够线程安全)
_stats = {"semantic_calls": 0, "semantic_hits": 0, "semantic_misses": 0}


def classify(
    query: str,
    labels: tuple[str, ...] | list[str] | set[str] | None = None,
) -> tuple[str, float] | None:
    """通用语义分类: 查询 vs 意图原型向量余弦相似度 (相对置信度判定)。

    labels: 仅在这些意图/技能里选 (如限制为已启用技能); None = 全部。
    判定: 最优 vs 次优 间距 ≥ config.INTENT_SEMANTIC_MARGIN 且 最优 ≥ FLOOR。
    返回 (label, score) 或 None (置信不足 → 交给大模型+工具层)。
    """
    from config import (
        INTENT_SEMANTIC_ENABLED,
        INTENT_SEMANTIC_FLOOR,
        INTENT_SEMANTIC_MARGIN,
    )

    if not INTENT_SEMANTIC_ENABLED:
        return None

    vectors = _proto_cache.vectors()
    if not vectors:
        return None

    import numpy as np
    from embedder import embed_query

    try:
        qv = np.asarray(embed_query(query))
    except Exception as e:
        logger.debug(f"Semantic classify embed failed: {e}")
        return None

    _stats["semantic_calls"] += 1
    # 相对置信度: bge 系模型绝对相似度挤在 ~0.7-0.9 窄带, 单一阈值区分不了
    # (实测 "帮我订张机票"=定义/0.72、"你好"=最新/0.79 均误判)。
    # 改用 最优 vs 次优 的间距(margin) + 绝对下限(floor) 双重判定。
    scored: list[tuple[float, str]] = []
    for label, vecs in vectors.items():
        if labels is not None and label not in labels:
            continue
        if not vecs:
            continue
        s = float(np.max(np.asarray(vecs) @ qv))  # 单位向量点积 = 余弦
        scored.append((s, label))

    best_score, best_label = 0.0, None
    for s, label in sorted(scored, reverse=True):
        if best_label is None:
            best_score, best_label = s, label
            continue
        # 与次优的间距足够大 且 绝对分数够高 → 采信; 否则交给大模型+工具层
        if best_score - s >= INTENT_SEMANTIC_MARGIN and best_score >= INTENT_SEMANTIC_FLOOR:
            _stats["semantic_hits"] += 1
            return (best_label, round(best_score, 4))
        break

    _stats["semantic_misses"] += 1
    return None


def classify_intent(question: str):
    """QA 意图语义兜底 (定义/对比/实操/最新/原理)。"""
    return classify(question, labels=QA_LABELS)


def classify_skill(
    task: str,
    active: tuple[str, ...] | list[str] | set[str] | None = None,
):
    """Agent 技能语义兜底 (仅限 active 中的技能; None = 全部技能)。"""
    labels = active if active is not None else SKILL_LABELS
    return classify(task, labels=labels)


def stats() -> dict:
    """语义层调用统计 (规则命中不算, 只统计语义兜底尝试)。"""
    return dict(_stats)


# ===== 自测 =====

def _pad(s: str, width: int) -> str:
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - w)


def _selftest() -> None:
    """打印规则 vs 语义在典型查询上的分类结果, 便于调阈值。"""
    from config import INTENT_KEYWORDS
    from agent.skills import SKILL_INTENT_KEYWORDS

    def rules_intent(q: str):
        ql = q.lower()
        for label, kws in INTENT_KEYWORDS.items():
            if any(k in ql for k in kws):
                return label
        return None

    def rules_skill(q: str):
        ql = q.lower()
        best, best_score = None, 0
        for name, kws in SKILL_INTENT_KEYWORDS.items():
            s = sum(1 for k in kws if k.lower() in ql)
            if s > best_score:
                best, best_score = name, s
        return best if best_score else None

    print(_pad("QA 查询 (规则 → 语义兜底)", 46))
    print("-" * 92)
    for q in [
        "GPT-4相对Claude的优势体现在哪些维度",
        "这个模型的改进点主要有哪些",
        "怎么把论文代码跑起来",
        "最新的大模型安全进展",
        "为什么量化会导致精度下降",
        "Transformer的核心组件有哪些",
        "帮我查一下某某的定义",
    ]:
        r = rules_intent(q)
        sem = classify_intent(q)
        sem_s = f"{sem[0]}/{sem[1]:.3f}" if sem else "None(→LLM)"
        print(f"{_pad(q, 40)} 规则={str(r or '✗'):<4s} 语义={sem_s}")

    print()
    print(_pad("技能查询 (规则 → 语义兜底)", 46))
    print("-" * 92)
    for q in [
        "帮我做一份大模型安全的文献综述",
        "调研联邦学习的研究现状",
        "帮我把实验结果写成论文",
        "解释知识蒸馏的原理",
        "分析近五年这个方向的发表趋势",
        "这个方向值不值得投入",
    ]:
        r = rules_skill(q)
        sem = classify_skill(q)
        sem_s = f"{sem[0]}/{sem[1]:.3f}" if sem else "None(→默认)"
        print(f"{_pad(q, 40)} 规则={str(r or '✗'):<18s} 语义={sem_s}")

    print(f"\nstats: {stats()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _selftest()
