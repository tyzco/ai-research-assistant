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
        # topic_id 现在是 12 位十六进制 (见 TestDeclaredConstraintsAreEnforced)
        assert t.validate({"urls": ["http://x"], "topic_id": "e9d9e4e4a9fa"}) == ""
        assert "格式不对" in t.validate({"urls": ["http://x"], "topic_id": "t1"})


class TestDeclaredConstraintsAreEnforced:
    """写在 schema 里却不校验的约束, 比不写更糟 —— 模型以为有, 实际没有,
    而错误会在下游以无关的形式出现。`pattern` 就是这样被漏掉的一项。"""

    def test_pattern_is_enforced(self):
        t = get_tool("get_topic_state")
        assert "pattern" in t.parameters["properties"]["topic_id"]
        assert t.validate({"topic_id": "e9d9e4e4a9fa"}) == ""      # 12 位十六进制
        assert "格式不对" in t.validate({"topic_id": "我的课题"})
        assert "格式不对" in t.validate({"topic_id": "TOO-SHORT"})

    def test_an_unconstrained_string_still_passes(self):
        """pattern 只对声明了它的参数生效, 不能误伤自由文本。"""
        t = get_tool("search_papers")
        assert t.validate({"query": "任意中文查询 fully free text"}) == ""

    def test_knowledge_base_name_explains_both_accepted_forms(self):
        """`table_name` 既接受 12 位 topic_id (自动解析) 也接受完整表名 ——
        这是模型无论如何猜不到的行为, 必须写在描述里。"""
        t = get_tool("agent_ask")
        desc = t.parameters["properties"]["table_name"]["description"]
        assert "topic_id" in desc
        assert "papers_" in desc


class TestThreeTierFallback:
    """三级兜底 —— 每级拦不同性质的失败。

    混在一起处理会让"能自愈的"和"必须停下的"走同一条路: 参数写错应该回灌给模型
    让它改 (有救), 而累计失败到阈值还继续下去, 模型会拿着一堆错误结果硬凑答案。
    """

    def _tool(self, retryable=True, calls=None):
        from agent.tool_registry import Tool

        state = calls if calls is not None else {"n": 0}

        def boom(**kw):
            state["n"] += 1
            raise RuntimeError("network timeout")

        return Tool("risky", "x", {"type": "object", "properties": {}}, boom,
                    retryable=retryable), state

    def test_level2_readonly_tools_retry(self):
        """可重放的操作: 超时重试是对的。"""
        import asyncio

        t, state = self._tool(retryable=True)
        out = asyncio.run(t.execute())
        assert state["n"] == 3          # 试了 3 次
        assert "重试3次后" in out

    def test_level2_side_effecting_tools_do_not_retry(self):
        """不可重放的操作: 失败即停 —— 超时不代表没生效。"""
        import asyncio

        t, state = self._tool(retryable=False)
        out = asyncio.run(t.execute())
        assert state["n"] == 1          # 只试了 1 次
        assert "不自动重试" in out
        # 必须把"先查状态"讲清楚, 否则模型会自己再调一遍, 绕过这层保护
        assert "先查询当前状态" in out

    def test_level1_a_parameter_error_carries_the_fix(self, tool):
        """一级: 回灌的不是"你错了", 而是"错在哪、应该是什么"。"""
        err = tool.validate({"query": "x", "limit": 9999})
        assert "最大为 50" in err              # 正确取值
        assert "参数结构" in err               # 完整 schema
        assert "重新调用" in err               # 明确的下一步

    # ── 行为测试: 真正驱动那个循环, 而不是检查源码里有没有某句话 ──

    def _drive_loop(self, monkeypatch, tool_call_builder, max_rounds=10):
        """跑一次 _run_agent, 用假的 LLM 反复要求调用工具。"""
        import asyncio
        import json as _json

        from agent import research_team as rt

        class _Fn:
            def __init__(self, name, args):
                self.name, self.arguments = name, args

        class _TC:
            def __init__(self, name, args):
                self.id, self.function = "call_1", _Fn(name, _json.dumps(args))

        class _Msg:
            def __init__(self, calls):
                self.tool_calls, self.content = calls, ""

            def model_dump(self, **kw):
                return {"role": "assistant", "content": self.content}

        class _Resp:
            def __init__(self, calls):
                self.choices = [type("C", (), {"message": _Msg(calls),
                                               "finish_reason": "tool_calls"})()]

        rounds = {"n": 0}
        seen_results = []

        async def fake_chat(**kw):
            rounds["n"] += 1
            # 把上一轮的工具结果记下来, 断言才有依据
            for m in kw.get("messages", []):
                if m.get("role") == "tool":
                    seen_results.append(m["content"])
            return _Resp([_TC(*tool_call_builder(rounds["n"]))])

        monkeypatch.setattr(rt, "chat_with_retry", fake_chat)

        team = rt.ResearchTeam(run_id="loop-test")
        out = asyncio.run(
            team._run_agent(
                "search_papers",
                {"name": "检索员", "role": "x", "layer": "upstream", "prompt": "x",
                 "tools": ["download_and_index"]},
            )
        )
        return out, rounds["n"], seen_results

    def test_level1_gives_one_correction_then_blocks_the_tool(self, monkeypatch):
        """一级: 回灌一次错误, 第二次就真正停用 —— 不是反复提示了事。

        参数错误是确定性的: 第二次写不对, 第三次也不会对。原实现只改提示文案、
        照旧执行, 循环一直跑到 8 轮上限 —— 那一级等于没有。
        """
        out, rounds, results = self._drive_loop(
            monkeypatch,
            lambda n: ("download_and_index",
                       {"urls": "not-a-list", "topic_id": "e9d9e4e4a9fa"}),
        )
        joined = "\n".join(results)
        assert "参数校验未通过" in joined     # 第 1 次: 带上正确 schema 让它改
        assert "本轮已停用" in joined          # 第 2 次: 真正停用
        assert rounds < 8                 # 没有一直烧到轮次上限

    def test_level3_terminates_the_task(self, monkeypatch):
        """三级: 累计失败到阈值必须终止 —— 否则模型会拿着一堆错误结果硬凑答案。"""
        out, rounds, results = self._drive_loop(
            monkeypatch, lambda n: ("citation_chain", {"paper_id": "!!not-hex!!"})
        )
        assert "已终止" in out
        assert "请勿据此继续判断" in out
        assert rounds < 8
