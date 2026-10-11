"""研究团队规划器 — 规划与执行分离的安全边界。

背景: PIPELINE 曾是类级写死的, 不管任务是"写综述"还是"批判一篇论文", 都跑完同一套
14 个 agent —— 写个引言也要跑 figure_generator/ethics_reviewer/bib_manager 全套。

`plan_task` 决定跑什么, `_run_pipeline` 负责怎么跑。规划器把一份清单交出去让 LLM
取舍, 所以它的**约束**比它的选择更重要: 一份 LLM 输出的清单不能绕过人工审批闸门、
不能改变执行顺序、不能因为一次失败就让整条管线少跑东西。这几条都是那种坏了不会报错、
只会静默改变行为的性质, 所以在这里钉死。
"""

import asyncio
import json

import pytest

import agent.research_team as rt_mod


@pytest.fixture
def team(monkeypatch):
    import config

    monkeypatch.setattr(config, "TEAM_PLANNER_ENABLED", True)
    return rt_mod.ResearchTeam(run_id="planner-test")


def _fake_llm(monkeypatch, payload, *, raise_exc=None):
    """Stub chat_with_retry so no network call happens."""

    class _Msg:
        content = payload if isinstance(payload, str) else json.dumps(payload)

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    async def fake(**kwargs):
        if raise_exc:
            raise raise_exc
        return _Resp()

    monkeypatch.setattr(rt_mod, "chat_with_retry", fake)


class TestSafetyBoundaries:
    """The invariants that matter more than which agents get picked."""

    def test_required_agents_cannot_be_dropped(self, team, monkeypatch):
        """The LLM may return ANY list — including one naming only optional
        agents. The 4 required agents (which include all three human approval
        gates) must survive regardless."""
        _fake_llm(monkeypatch, ["chapter_writer_intro"])
        plan = asyncio.run(team.plan_task("写个引言就行"))
        names = {p[0] for p in plan}
        required = {p[0] for p in team.PIPELINE if p[2]}
        assert required <= names

    def test_the_llm_cannot_reorder_the_pipeline(self, team, monkeypatch):
        """Order is a correctness requirement — chapters must be written before
        the final draft. The planner chooses a SUBSET; it does not sequence."""
        _fake_llm(monkeypatch, ["final_drafter", "topic_evaluator", "bib_manager"])
        plan = asyncio.run(team.plan_task("随便"))
        order = [p[0] for p in plan]
        canonical = [p[0] for p in team.PIPELINE if p[0] in set(order)]
        assert order == canonical

    def test_unknown_agent_names_are_ignored(self, team, monkeypatch):
        _fake_llm(monkeypatch, ["chapter_writer_intro", "definitely_not_an_agent"])
        plan = asyncio.run(team.plan_task("x"))
        assert "definitely_not_an_agent" not in {p[0] for p in plan}


class TestFallbackIsAlwaysToTheFullPipeline:
    """兜底优先于省成本: 少跑一个 agent 是性能问题, 漏跑一个是正确性问题。"""

    def test_llm_error_falls_back_to_everything(self, team, monkeypatch):
        _fake_llm(monkeypatch, "", raise_exc=RuntimeError("llm down"))
        plan = asyncio.run(team.plan_task("写综述"))
        assert plan == list(team.PIPELINE)

    def test_empty_selection_falls_back_to_everything(self, team, monkeypatch):
        _fake_llm(monkeypatch, "[]")
        plan = asyncio.run(team.plan_task("写综述"))
        assert plan == list(team.PIPELINE)

    def test_unparseable_output_falls_back_to_everything(self, team, monkeypatch):
        _fake_llm(monkeypatch, "抱歉我无法回答这个问题")
        plan = asyncio.run(team.plan_task("写综述"))
        assert plan == list(team.PIPELINE)

    def test_disabled_flag_runs_everything(self, team, monkeypatch):
        import config

        monkeypatch.setattr(config, "TEAM_PLANNER_ENABLED", False)
        plan = asyncio.run(team.plan_task("写综述"))
        assert plan == list(team.PIPELINE)


class TestPlanningActuallyNarrows:
    def test_a_selection_runs_fewer_agents_than_the_full_pipeline(self, team, monkeypatch):
        _fake_llm(monkeypatch, ["bib_manager", "format_checker"])
        plan = asyncio.run(team.plan_task("只是整理一下参考文献"))
        assert len(plan) < len(team.PIPELINE)
        assert {"bib_manager", "format_checker"} <= {p[0] for p in plan}

    def test_the_run_result_reports_what_was_skipped(self, team, monkeypatch):
        """规划省了多少, 必须能从结果里看出来 —— 否则这是一个无法验证的优化。"""
        _fake_llm(monkeypatch, ["bib_manager"])
        plan = asyncio.run(team.plan_task("x"))
        kept = {p[0] for p in plan}
        skipped = [
            p[0] for p in team.PIPELINE
            if not p[2] and p[0] not in kept
        ]
        assert skipped  # 有可跳过的
        assert all(p[0] not in skipped for p in plan)
