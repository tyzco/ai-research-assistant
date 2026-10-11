"""工具参数校验 — 模型写错的参数必须被拦下, 并且**告诉它哪里错了**。

两类问题:
  1. `Tool.execute` 此前只查 `required`。类型、区间、枚举一概不看, 于是
     `max_results: 99999` 和 `direction: "随便"` 被原样接受, 由工具或下游 API
     再报一个跟真实原因无关的错。
  2. Function Calling 循环里 `except JSONDecodeError: args = {}` —— 参数 JSON
     坏掉时静默当空参数。模型自始至终不知道自己写坏了, 也就没有机会重写。

校验的价值不只是"拦住", 更是把"哪里错、应该是什么"回灌给模型 ——
只返回 False 是白白浪费了模型自我修正的那一次机会。
"""

import json

import pytest

from agent.tool_registry import Tool, get_tool


def _echo(**kwargs):
    return kwargs


@pytest.fixture
def tool():
    return Tool(
        "demo",
        "测试工具",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "direction": {"type": "string", "enum": ["both", "forward", "backward"]},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            },
            "required": ["query"],
        },
        _echo,
    )


class TestValidationCatchesWhatRequiredOnlyMissed:
    def test_a_missing_required_parameter(self, tool):
        assert "缺少必填参数 `query`" in tool.validate({})

    def test_a_whitespace_only_required_parameter(self, tool):
        assert "不能为空字符串" in tool.validate({"query": "   "})

    def test_wrong_type(self, tool):
        err = tool.validate({"query": "x", "limit": "ten"})
        assert "类型不对" in err and "integer" in err

    def test_out_of_range_is_rejected_with_the_bounds(self, tool):
        err = tool.validate({"query": "x", "limit": 9999})
        assert "最大为 50" in err
        assert "9999" in err

    def test_below_minimum(self, tool):
        assert "最小为 1" in tool.validate({"query": "x", "limit": 0})

    def test_enum_violation_lists_the_allowed_values(self, tool):
        err = tool.validate({"query": "x", "direction": "sideways"})
        assert "both" in err and "forward" in err

    def test_array_length_cap(self, tool):
        assert "最多 3 项" in tool.validate({"query": "x", "tags": ["a", "b", "c", "d"]})

    def test_valid_args_pass(self, tool):
        assert tool.validate(
            {"query": "x", "limit": 10, "direction": "both", "tags": ["a"]}
        ) == ""

    def test_bools_are_not_integers(self, tool):
        """bool 是 int 的子类 —— `limit=True` 会悄悄通过 integer 校验。"""
        assert "类型不对" in tool.validate({"query": "x", "limit": True})


class TestTheErrorIsUsefulToTheModel:
    """拦截的意义在于模型能据此重写, 所以错误必须带上它需要的信息。"""

    def test_the_error_includes_the_expected_schema(self, tool):
        err = tool.validate({"query": "x", "limit": 9999})
        assert "参数结构" in err
        assert "maximum" in err  # 模型能看到正确取值

    def test_the_error_says_it_should_retry(self, tool):
        assert "重新调用" in tool.validate({"query": "x", "limit": 9999})

    def test_execute_returns_the_same_error_instead_of_running(self, tool):
        import asyncio

        out = asyncio.run(tool.execute(query="x", limit=99999))
        assert out.startswith("❌")
        assert "最大为 50" in out
        # 没有真的执行 —— 否则返回的是 kwargs 的 JSON
        assert '"limit": 99999' not in out


class TestRealToolsHaveUsableSchemas:
    def test_a_registered_tool_validates_end_to_end(self):
        t = get_tool("download_and_index")
        assert t is not None
        assert "缺少必填参数" in t.validate({"urls": ["http://x"]})
        assert t.validate({"urls": ["http://x"], "topic_id": "t1"}) == ""
