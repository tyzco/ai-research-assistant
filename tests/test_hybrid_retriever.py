"""hybrid_retriever 单元测试 — 自适应 BM25 权重推理 (纯函数, 无模型加载)。"""
from hybrid_retriever import AdaptiveHybridRetriever


def _weight(query: str) -> float:
    # _compute_adaptive_weight 不引用 self, 用类直接调用
    return AdaptiveHybridRetriever._compute_adaptive_weight(None, query)


def test_short_query_biases_keyword():
    # 短查询 (<15字) → BM25 权重更高
    assert _weight("神经网络") > _weight("请解释一下神经网络在自然语言处理中的应用原理")


def test_question_form_biases_semantic():
    # 问句形式 → BM25 权重降低
    base = _weight("神经网络")
    assert _weight("什么是神经网络") < base


def test_english_acronym_biases_keyword():
    # 含英文缩写 → BM25 权重更高
    assert _weight("CNN 结构") > _weight("卷积神经网络的结构是什么")


def test_weight_in_valid_range():
    # 权重始终在 [0.05, 0.95]
    for q in ["短", "什么是", "LSTM 和 GRU 区别", "这是一个非常长的自然语言问句用来测试边界"]:
        w = _weight(q)
        assert 0.05 <= w <= 0.95, f"权重越界: {q} -> {w}"
