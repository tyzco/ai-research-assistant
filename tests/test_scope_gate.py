"""scope_gate 学术白名单闸门单元测试 — 只测确定性规则路径 (不触发嵌入模型)。"""
from scope_gate import gate_query

# 这些用例全部命中 L0/L1 规则层, 不进入语义层 (避免加载嵌入模型)


def test_academic_keyword_pass():
    for q in [
        "帮我写一篇关于图神经网络的综述",
        "帮我分析这篇论文的实验设计",
        "什么是transformer",
        "如何训练一个扩散模型",
        "论文引用格式怎么写",
    ]:
        allowed, _, level = gate_query(q)
        assert allowed, f"学术请求被误拒: {q} (level={level})"


def test_non_academic_keyword_reject():
    for q in ["今天股票涨了吗", "帮我写简历", "怎么入侵别人的网站", "帮我写一个python爬虫", "帮我做饭", "今天天气怎么样"]:
        allowed, _, level = gate_query(q)
        assert not allowed, f"非学术请求被放行: {q} (level={level})"


def test_continuation_short_pass():
    for q in ["继续", "嗯", "好的", "帮我写", "补充"]:
        allowed, _, level = gate_query(q)
        assert allowed, f"续接短语被误拒: {q} (level={level})"


def test_empty_reject():
    allowed, _, level = gate_query("")
    assert not allowed
    assert level == "empty"


def test_whitespace_only_reject():
    allowed, _, _ = gate_query("   ")
    assert not allowed
