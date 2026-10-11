"""Pi-Agent 风格的主循环 — Hook 驱动 + 上下文工程 + 工具管理。

核心架构 (参考 pi-agent):
  单次任务(trace) → 多轮模型调用(turn)
  → 每轮经过 10 个 hook 干预节点
  → 上下文压缩 + 工具输出截断 + 错误恢复

使用:
    from agent.pi_agent import PiAgent
    agent = PiAgent(max_steps=20, goal="文献综述")
    result = await agent.run(task="调研人脸识别的最新进展")
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from config import (
    AGENT_MAX_TOKENS,
    AGENT_MAX_STEPS,
    AGENT_MAX_CONSECUTIVE_FAILURES,
    AGENT_REPEAT_CALL_LIMIT,
    AGENT_PLAN_ENABLED,
    AGENT_PLAN_MIN_CHARS,
    AGENT_REFLEXION_ENABLED,
    AGENT_TEMPERATURE,
    DEEPSEEK_MODEL,
)
from llm_client import chat_with_retry

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """单轮状态。"""

    index: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0
    tool_name: str = ""
    tool_success: bool = False
    elapsed_ms: float = 0.0


@dataclass
class TraceState:
    """任务级别状态。"""

    goal: str = ""
    turns: list[TurnState] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    start_time: float = 0.0
    total_tokens: int = 0
    # 重复动作检测: 最近若干次 (工具名, 参数) 指纹。ReAct 最典型的失败模式是
    # 模型反复用同样参数调同一个工具却不推进 —— 只靠 max_steps 要烧满 15 轮才发现。
    call_fingerprints: list[str] = field(default_factory=list)
    repeat_hits: int = 0
    # 错误累积: 连续失败计数 (成功即清零)。用于在错误路径上及时止损。
    consecutive_failures: int = 0
    # 一级兜底: 每个工具的参数错误次数。参数错是**确定性**的 —— 模型改不对说明它
    # 没理解工具契约, 再试只是烧步数 (与"失败兜底"分级的思路一致:
    # 一级给一次修正机会, 二级靠重试层处理瞬时故障, 三级才整体熔断)。
    param_failures: dict = field(default_factory=dict)


class PiAgent:
    """Hook 驱动的 Agent 主循环。

    pi-agent 设计:
      - trace = 单次任务 (从用户输入到最终输出)
      - turn = 一次模型调用 + 可能的一次工具执行
      - 10 个 hook 节点覆盖完整生命周期
    """

    def __init__(
        self,
        max_steps: int = AGENT_MAX_STEPS,
        max_time_seconds: int = 300,
        goal: str = "",
        enable_tool_check: bool = True,
        enable_context_compaction: bool = True,
        skills: list[str] | None = None,
        run_id: str = "",
        step_cb=None,
        enable_reflexion: bool | None = None,
        enable_plan: bool | None = None,
        kb_context: str = "",
        topic_id: str = "",
        owner: str = "",
    ):
        # 发起者身份 (登录 user_id, 匿名则客户端 IP)。人工确认闸靠它校验:
        # /agent/pi/approve 只接受与 run 发起者相同身份的请求 —— 否则任何人都能
        # 拿一个 run_id 批准 agent 暂停的危险动作。
        self.owner = owner
        self.max_steps = max_steps
        self.max_time = max_time_seconds
        self.goal = goal
        self.enable_tool_check = enable_tool_check
        self.enable_compaction = enable_context_compaction
        # DeepSeek 审查: 显式技能名覆盖 auto 意图选择 (None = 按任务自动激活)
        self._skills_override = skills
        self.run_id = run_id or f"pi_{uuid.uuid4().hex[:12]}"
        # 步骤流回调 (流式 SSE: 每步工具完成即推送事件, 供 /agent/pi/stream 使用)
        self._step_cb = step_cb
        # 反思重试闭环: 直答后自批判, 不足则带批判重试一轮 (有界, 最多 1 次)
        self.enable_reflexion = (
            AGENT_REFLEXION_ENABLED if enable_reflexion is None else enable_reflexion
        )
        self._reflexed = False
        # 可选 plan 阶段 (规划层): 长任务先拆步骤再执行, 防纯 ReAct 跑偏
        self.enable_plan = AGENT_PLAN_ENABLED if enable_plan is None else enable_plan
        # 当前课题知识库上下文 (限制 agent 只查当前课题的表, 不跨库)
        self.kb_context = kb_context
        # 当前课题 id (用于把 agent 发现/工具调用持久化到课题记忆 — 接线 add_finding/add_tool_call)
        self.topic_id = topic_id

        # 延迟加载
        self._hooks = None
        self._ctx_engine = None
        self._tool_mgr = None

        # 暂停/恢复状态 (人工确认门, 合并自 langgraph)
        self._trace: Optional[TraceState] = None
        self._messages: list[dict] = []
        self._state: dict = {}
        self._tools_schema: list[dict] = []
        self._pending_confirm: Optional[
            dict
        ] = None  # {tool_name, tool_args, tool_call_id}
        # 同一轮里多个工具都要人工确认时, 除当前这个外的其余排队 (含仍未处理的兄弟 tool_call)。
        # 逐个向客户端抛出, 而不是一次只认第一个、其余静默丢弃。
        self._pending_queue: list[dict] = []

    @property
    def hooks(self):
        if self._hooks is None:
            from agent.hooks import get_hooks

            self._hooks = get_hooks()
        return self._hooks

    @property
    def ctx(self):
        if self._ctx_engine is None:
            from agent.context_engine import get_context_engine

            self._ctx_engine = get_context_engine()
        return self._ctx_engine

    @property
    def tool_mgr(self):
        if self._tool_mgr is None:
            from agent.tool_manager import get_tool_manager

            self._tool_mgr = get_tool_manager()
        return self._tool_mgr

    def _fire_hook(self, point, state, payload: str = ""):
        """触发 hook，异常隔离 (DeepSeek 审查 #4): 单个 hook 崩溃不中断 agent。"""
        try:
            from agent.hooks import get_hooks

            return get_hooks().fire(point, state, payload)
        except Exception as e:
            logger.warning(f"Hook {point} failed: {e}")
            from agent.hooks import HookResult

            return HookResult()  # 默认无阻断

    async def run(self, task: str) -> dict:
        """执行完整的 agent trace (人工确认暂停后可由 resume 续跑)。"""
        # === 技能解析: 显式指定 > 意图自动选择 (DeepSeek 审查 #2/#3/#4) ===
        from agent.skills import get_skills, schemas_for_tools

        skill_reg = get_skills()
        if self._skills_override:
            selected = [
                n for n in self._skills_override if skill_reg.get(n) is not None
            ]
            if not selected:
                selected = skill_reg.select_for_task(task)
        else:
            selected = skill_reg.select_for_task(task)
        skill_prompt = skill_reg.prompt_for_skills(selected)
        # 兜底: 路由证据弱时不收窄工具集 (路由错会丢建库/导出等能力, 见 skills.tools_for_task)
        tool_names = skill_reg.tools_for_task(task, selected)
        self._tools_schema = schemas_for_tools(tool_names)

        self._trace = TraceState(goal=self.goal or task[:80], start_time=time.time())
        self._messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(skill_prompt, task),
            },
            {"role": "user", "content": task},
        ]

        # === HOOK: trace_start ===
        hook_result = self._fire_hook("trace_start", {"task": task, "goal": self.goal})
        if hook_result.block:
            return {
                "success": False,
                "result": hook_result.block_reason,
                "trace": self._trace,
            }
        if hook_result.inject:
            self._messages.extend(hook_result.inject)

        # === 可选 plan 阶段: 长任务先规划 (防跑偏), 短问答跳过 (省延迟) ===
        if self.enable_plan and len(task.strip()) >= AGENT_PLAN_MIN_CHARS:
            _plan_ret = await self._plan(task)
            if _plan_ret:
                _p, _p_rc = _plan_ret
                _pm = {
                    "role": "assistant",
                    "content": f"任务计划：\n{_p}",
                }
                if _p_rc:
                    _pm["reasoning_content"] = _p_rc  # 回传, 否则后续调用 400
                self._messages.append(_pm)
                # Plan 模式 (人工介入的"方向闸"): 计划生成后**暂停等人工确认**再执行 ——
                # 长任务的错方向若跑完 20 步才发现, 代价远高于先看一眼计划。
                # 复用工具确认门 (_pending_confirm + /agent/pi/approve), 用 type 区分两种暂停。
                from config import AGENT_PLAN_REQUIRE_APPROVAL

                if AGENT_PLAN_REQUIRE_APPROVAL:
                    self._pending_confirm = {
                        "type": "plan",
                        "plan": _p,
                        "run_id": self.run_id,
                    }
                    self._state = {
                        "messages": self._messages,
                        "max_steps": self.max_steps,
                        "steps": 0,
                    }
                    self._persist_pending()
                    self._fire_hook(
                        "trace_end", {"result": "paused_for_plan", "task": task}
                    )
                    return {
                        "success": True,
                        "paused": True,
                        "need_plan_approval": True,
                        "plan": _p,
                        "run_id": self.run_id,
                        "result": _p,
                    }

        self._state = {
            "messages": self._messages,
            "max_steps": self.max_steps,
            "steps": 0,
            # === 统一 state: 意图 + 实体 (此前只存 messages/steps, 意图与实体散落在各处) ===
            # intent: 技能路由结果 + 规则/语义意图 (三层漏斗的输出)
            # entities: 从任务文本抽取的结构化槽位 (零额外 LLM 调用, 纯规则)
            "intent": _describe_intent(task, selected),
            "entities": _extract_entities(task),
            "skills": selected,
        }
        result = await self._loop()
        # P3 持久技能记忆: 成功任务记录做法 (跨会话复用成功经验)
        try:
            if result.get("success"):
                from agent_skills import record_success

                trace = result.get("trace") or {}
                record_success(
                    task,
                    trace.get("tools_used") or [],
                    str(result.get("result", ""))[:150],
                    user_id=self.owner,
                )
        except Exception:
            pass
        # 接线 add_finding/add_tool_call: 把 agent 的发现/工具调用持久化到课题记忆 (之前从未调用)
        if self.topic_id:
            try:
                from conversation_memory import get_memory

                _mem = get_memory(self.topic_id)
                _tr = result.get("trace") or {}
                for _t in _tr.get("tools_used") or []:
                    _mem.add_tool_call(_t, "")
                for _kf in _tr.get("key_findings") or []:
                    _mem.add_finding(str(_kf)[:300], "agent")
            except Exception:
                pass
        self._verify_high_risk(result)
        return result

    def _verify_high_risk(self, result: dict) -> None:
        """系统层硬性核验 (不由 LLM 自觉) —— 对应"高风险任务不允许出错"这一 Agent 适配前提。

        写作/初稿类任务的"引用真实性"是学术高风险项 (幻觉引文=学术不端), 不能只靠
        system prompt 提示「必须核验」而依赖 LLM 自觉。此处按 trace 里的实际工具调用判定:
        若未调用 check_draft_quality / check_factual_support, 就在结果上**显式标注未核验**
        (而非默默放行), 提示人工复核。
        """
        if not result.get("success"):
            return
        try:
            _tr = result.get("trace") or {}
            _tools = _tr.get("tools_used") or []
            _ctx = (self.goal or "") + str(self._state.get("task", ""))
            _is_writing = any(
                k in _ctx for k in ("写", "综述", "初稿", "论文", "大纲")
            )
            if not _is_writing:
                return
            _required = ("check_draft_quality", "check_factual_support")
            _missing = [t for t in _required if t not in _tools]
            if _missing:
                result["unverified"] = _missing
                result["result"] = str(result.get("result", "")) + (
                    f"\n\n---\n⚠️ **系统核验提示**：本次写作未调用 `{', '.join(_missing)}`——"
                    f"引用真实性与初稿质量**未经系统核验**。请人工复核关键论断与引用后再使用。"
                )
                logger.warning(f"High-risk writing unverified: missing {_missing}")
        except Exception as e:
            logger.warning(f"_verify_high_risk failed: {e}")

    async def _plan(self, task: str) -> tuple[str, str] | None:
        """可选 plan 阶段 (规划层): 长任务先让 LLM 拆 3-6 步计划, 防纯 ReAct 跑偏。

        返回 (计划, reasoning_content) — 后者必须随 assistant 消息回传 (DeepSeek 契约)。
        """
        try:
            resp = await chat_with_retry(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是任务规划器。把用户任务拆成 3-6 个有序步骤, "
                            "每步一句, 用编号列表输出。只输出计划, 不执行、不解释。"
                        ),
                    },
                    {"role": "user", "content": task},
                ],
                temperature=0.1,
                max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}},
            )
            plan = (resp.choices[0].message.content or "").strip()
            if not plan:
                return None
            rc = getattr(resp.choices[0].message, "reasoning_content", None) or ""
            return (plan[:500], rc)
        except Exception as e:
            logger.debug(f"Plan stage failed: {e}")
            return None

    async def _reflexion_critique(self, messages: list[dict]) -> str:
        """轻量自批判: 评估回答是否完整满足任务。返回批判 (非空=需修订) 或 '' (通过)。"""
        try:
            resp = await chat_with_retry(
                model=DEEPSEEK_MODEL,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "严格评估你上面的回答是否完整、准确、充分满足任务要求。"
                            "若已完整满足, 只返回: PASS。若不完整或有遗漏, "
                            "返回 2-3 句具体缺什么、怎么改进 (不要客套)。"
                        ),
                    }
                ],
                temperature=0.1,
                max_tokens=200,
                extra_body={"thinking": {"type": "disabled"}},
            )
            critique = (resp.choices[0].message.content or "").strip()
            if critique.upper().startswith("PASS"):
                return ""
            return critique[:300]
        except Exception as e:
            logger.debug(f"Reflexion critique failed: {e}")
            return ""

    async def _loop(self) -> dict:
        """主循环 (人工确认处可暂停, resume 后继续)。"""
        trace = self._trace
        messages = self._messages
        state = self._state
        final_result = ""

        for turn_idx in range(self.max_steps):
            # === 超时检查 ===
            if time.time() - trace.start_time > self.max_time:
                final_result = f"任务超时 ({self.max_time}s)"
                break

            turn = TurnState(index=turn_idx)
            t_turn_start = time.time()

            # === HOOK: turn_start ===
            hr = self._fire_hook("turn_start", state)
            if hr.block:
                final_result = hr.block_reason
                break

            # === 上下文压缩检查 ===
            if self.enable_compaction and self.ctx.should_compact(messages):
                # DeepSeek 审查 #2/#9: compact_history 保留 messages[-4:],
                # 实际移除的是 messages[:-4]。按真实移除量扣减并 clamp ≥ 0。
                removed = messages[:-4] if len(messages) > 4 else []
                messages = self.ctx.compact_history(
                    messages,
                    goal=trace.goal,
                    tools_used=trace.tools_used,
                    key_findings=trace.key_findings,
                )
                if removed:
                    trace.total_tokens = max(
                        0,
                        trace.total_tokens - self.ctx.estimate_messages_tokens(removed),
                    )
                state["messages"] = messages
                self._messages = messages
                logger.info(f"Context compacted: {len(messages)} messages after")

            # === 工具 schema (技能裁剪, before_model hook 可再覆盖) ===
            tools = self._tools_schema

            # === HOOK: before_model ===
            hr = self._fire_hook("before_model", state)
            if hr.block:
                final_result = hr.block_reason
                break
            if hr.tools is not None:
                tools = hr.tools  # intersection-merge already applied
            if hr.messages is not None:
                messages = hr.messages
                state["messages"] = messages
                self._messages = messages

            # === LLM 调用 ===
            try:
                resp = await chat_with_retry(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=AGENT_TEMPERATURE,
                    max_tokens=AGENT_MAX_TOKENS,
                    # 修复: 主循环缺 thinking disabled → flash 产出 reasoning_content → 下一轮 400。
                    # 全链路一致禁用 thinking (plan/反思/续写都已禁), 否则推理模型契约要求回传 reasoning_content。
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception as e:
                logger.error(f"LLM call failed at turn {turn_idx}: {e}")
                final_result = f"LLM 调用失败: {e}"
                break

            msg = resp.choices[0].message
            turn.model_input_tokens = resp.usage.prompt_tokens if resp.usage else 0
            turn.model_output_tokens = resp.usage.completion_tokens if resp.usage else 0
            trace.total_tokens += turn.model_input_tokens + turn.model_output_tokens

            # === HOOK: after_model ===
            hr = self._fire_hook(
                "after_model", state, json.dumps({"has_tools": bool(msg.tool_calls)})
            )
            if hr.block:
                final_result = hr.block_reason
                break

            if msg.tool_calls:
                # === 工具执行路径 ===
                # DeepSeek 审查 #3: 全量处理 tool_calls, 不再丢弃多余调用
                # DeepSeek 审查 #5: assistant 消息只追加一次, 每条工具结果都回填
                # 修复: 追加 dict 而非原始 ChatCompletionMessage 对象 —
                # 原 append(msg) 使 messages 混入 pydantic 对象, 下一轮 context_engine
                # estimate_messages_tokens 的 m.get("content") 崩 AttributeError
                # (仅在 agent 发起工具调用时触发, 直答轮不触发)。
                # 回传 reasoning_content (DeepSeek 推理模型契约: 上一轮有 thinking 必须原样回传, 否则 400)
                _rc = getattr(msg, "reasoning_content", None)
                _am = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                if _rc:
                    _am["reasoning_content"] = _rc
                messages.append(_am)
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    # 硬上限: check_draft_quality 最多 2 次, 之后阻止并强制交付 (防 10 次自检烧 token)
                    if tool_name == "check_draft_quality":
                        self._quality_checks = getattr(self, "_quality_checks", 0) + 1
                        if self._quality_checks > 2:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": "已自检 2 次。请不要再调用 check_draft_quality，直接交付最终初稿（可标注仍未满足的项）。",
                                }
                            )
                            state["steps"] += 1
                            continue
                    # 硬上限: agent_ask 每 agent 最多 3 次 (防简单问题过度查库, 慢+乱)
                    if tool_name == "agent_ask":
                        self._ask_checks = getattr(self, "_ask_checks", 0) + 1
                        if self._ask_checks > 3:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": "已查询知识库 3 次，已有足够信息。请直接基于已有检索结果回答，不要继续查询。",
                                }
                            )
                            state["steps"] += 1
                            continue
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    turn.tool_name = tool_name
                    tool_result = None  # None = 待执行

                    # === 工具参数校验 (pi-agent 风格自动修正) ===
                    from agent.tool_registry import get_tool

                    tool_def = get_tool(tool_name)
                    if tool_def and self.enable_tool_check:
                        is_valid, corrected, err = self.tool_mgr.validate_params(
                            tool_name, tool_args, tool_def.parameters
                        )
                        if is_valid:
                            tool_args = corrected
                        else:
                            _n = trace.param_failures.get(tool_name, 0) + 1
                            trace.param_failures[tool_name] = _n
                            from config import AGENT_PARAM_CORRECTION_BUDGET

                            if _n > AGENT_PARAM_CORRECTION_BUDGET:
                                # 一级兜底用尽: 参数错误改不对通常不是"再试一次就好",
                                # 而是模型没理解工具契约 —— 明确禁止再调, 让它换路。
                                logger.warning(
                                    f"param correction budget exhausted for {tool_name} "
                                    f"({_n} 次参数错误)"
                                )
                                try:
                                    from monitor import metrics as _mx

                                    _mx.record_degradation("param_budget_exhausted")
                                except Exception:
                                    pass
                                tool_result = (
                                    f"⛔ 已停止调用 {tool_name}: 连续 {_n} 次参数错误。"
                                    f"说明该工具的调用方式没被正确理解 —— "
                                    f"请改用其他方法完成任务，不要再调用这个工具。\n"
                                    f"最后一次错误: {err}"
                                )
                            else:
                                tool_result = self.tool_mgr.format_error_for_llm(
                                    tool_name, err, tool_args
                                )
                            turn.tool_success = False

                    # === 人工确认门 (合并自 langgraph) ===
                    if (
                        tool_result is None
                        and tool_def is not None
                        and tool_def.need_confirm
                    ):
                        self._pending_confirm = {
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "tool_call_id": tc.id,
                        }
                        # 收集本轮剩余待确认工具: 同一 assistant 消息里当前 tc 之后的兄弟
                        # tool_call, 以及后续消息里需要确认的 (此处只有兄弟, 因为本轮回传
                        # 了全部 tool_calls 的响应才会继续)。逐个排队依次抛给客户端。
                        self._pending_queue = self._collect_remaining_confirms(
                            msg, tc.id
                        )
                        self._messages = messages
                        self._state = state
                        self._persist_pending()
                        self._fire_hook(
                            "trace_end",
                            {"result": "paused_for_confirm", "turns": len(trace.turns)},
                        )
                        return {
                            "success": True,
                            "paused": True,
                            "run_id": self.run_id,
                            "need_confirm": self._pending_confirm,
                            "result": f"需要人工确认执行工具: {tool_name}",
                            "trace": {
                                "goal": trace.goal,
                                "turns": len(trace.turns),
                                "tools_used": trace.tools_used,
                                "total_tokens": trace.total_tokens,
                                "elapsed": round(time.time() - trace.start_time, 1),
                                "steps": [
                                    {
                                        "turn": t.index + 1,
                                        "tool": t.tool_name or "llm",
                                        "success": t.tool_success,
                                        "tokens": t.model_input_tokens
                                        + t.model_output_tokens,
                                        "elapsed_ms": round(t.elapsed_ms),
                                    }
                                    for t in trace.turns
                                ],
                            },
                            "steps": state["steps"],
                        }

                    # === HOOK: before_tool + 安全检查 ===
                    if tool_result is None:
                        hr = self._fire_hook(
                            "before_tool",
                            state,
                            json.dumps({"tool": tool_name, "args": tool_args}),
                        )
                        if hr.block:
                            tool_result = f"工具 {tool_name} 被安全策略拦截: {hr.block_reason}"
                            turn.tool_success = False
                        elif self.enable_tool_check:
                            safe, reason = self.tool_mgr.security_check(
                                tool_name, tool_args
                            )
                            if not safe:
                                tool_result = f"安全检查未通过: {reason}"
                                turn.tool_success = False

                    # === 重复动作检测 (死循环第三道防线) ===
                    # 轮次限制(max_steps) 与超时兜底(max_time) 都是"烧到上限才停";
                    # 这里在**动作层面**提前发现原地打转: 同工具+同参反复调用 → 直接把
                    # 观察结果换成警示, 让模型换路, 而不是等它把步数耗光。
                    if tool_result is None:
                        _fp = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)[:200]}"
                        _recent = trace.call_fingerprints[-6:]
                        _repeats = _recent.count(_fp)
                        trace.call_fingerprints.append(_fp)
                        if _repeats >= AGENT_REPEAT_CALL_LIMIT - 1:
                            trace.repeat_hits += 1
                            logger.warning(
                                f"repeat action detected: {tool_name} ×{_repeats + 1} "
                                f"(goal={trace.goal[:40]})"
                            )
                            try:
                                from monitor import metrics as _mx

                                _mx.record_degradation("agent_repeat_action")
                            except Exception:
                                pass
                            tool_result = (
                                f"⚠️ 检测到重复动作：你已连续 {_repeats + 1} 次用**相同参数**调用 "
                                f"{tool_name}，结果不会变化。请改用不同参数/其他工具，"
                                f"或直接基于已有信息给出最终答复。"
                            )
                            turn.tool_success = False
                            turn.tool_name = tool_name

                    # === 执行工具 ===
                    if tool_result is None and tool_def is not None:
                        # DeepSeek 审查 #6: 用剩余时间做超时保护, 工具卡住可中止
                        try:
                            remaining = self.max_time - (time.time() - trace.start_time)
                            if remaining <= 0:
                                tool_result = "任务超时, 跳过工具执行"
                                turn.tool_success = False
                            else:
                                # 瞬时故障 (超时/连接/限流/5xx) 自动退避重试; 参数类错误不重试
                                _ok, _res = await self.tool_mgr.execute_with_retry(
                                    tool_name,
                                    tool_def,
                                    tool_args,
                                    timeout=min(remaining, 60),
                                )
                                if _ok:
                                    raw_result = _res
                                    turn.tool_success = True
                                    trace.tools_used.append(tool_name)
                                else:
                                    raw_result = self.tool_mgr.format_error_for_llm(
                                        tool_name, str(_res), tool_args
                                    )
                                    turn.tool_success = False
                        except Exception as e:
                            raw_result = self.tool_mgr.format_error_for_llm(
                                tool_name, str(e), tool_args
                            )
                            turn.tool_success = False

                        # === 工具输出截断 (pi-agent 风格) ===
                        if tool_result is None:
                            tool_result = str(raw_result)
                            truncated, filepath = self.ctx.truncate_tool_output(
                                tool_name, tool_result
                            )
                            if filepath:
                                logger.info(
                                    f"Tool output truncated: {tool_name} "
                                    f"({len(tool_result)} → {len(truncated)} chars)"
                                )
                    elif tool_result is None:
                        tool_result = f"未知工具: {tool_name}"
                        turn.tool_success = False
                    turn.tool_result = str(tool_result)[:200]  # 存预览供 key_findings

                    # === 错误累积熔断 (ReAct 第三类故障模式) ===
                    # 单次失败由模型自行纠错是对的; 但**连续**失败说明它不是"再试一下就能过",
                    # 而是路线本身错了 —— 此时继续烧步数只会把错误一路累积进上下文,
                    # 最后交出一份建立在失败之上的答案。这里及时止损, 如实报告。
                    if turn.tool_success:
                        trace.consecutive_failures = 0
                    else:
                        trace.consecutive_failures += 1
                        if trace.consecutive_failures >= AGENT_MAX_CONSECUTIVE_FAILURES:
                            logger.warning(
                                f"错误累积熔断: 连续 {trace.consecutive_failures} 次工具失败 "
                                f"(goal={trace.goal[:40]})"
                            )
                            try:
                                from monitor import metrics as _mx

                                _mx.record_degradation("agent_error_accumulation")
                            except Exception:
                                pass
                            final_result = (
                                f"已连续 {trace.consecutive_failures} 次工具调用失败，"
                                f"为避免在错误路径上继续消耗，已提前终止。\n\n"
                                f"最后一次失败: {str(tool_result)[:300]}\n\n"
                                f"建议：换一个更具体的任务描述，或先确认知识库/网络状态后重试。"
                            )
                            break

                    # === HOOK: after_tool ===
                    self._fire_hook(
                        "after_tool",
                        state,
                        json.dumps({"tool": tool_name, "success": turn.tool_success}),
                    )

                    # 记录调用
                    self.tool_mgr.record_call(
                        tool_name, tool_args, str(tool_result)[:200], turn.tool_success
                    )

                    # DeepSeek 审查 #5: 校验/拦截/失败都回填结果, 避免对话停滞
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result)[:3000],
                        }
                    )
                    # DeepSeek 审查 #1: 任何路径都推进 steps, 防死循环
                    state["steps"] += 1
            else:
                # === 直接回答 (无工具调用) — 流式生成, 逐字推送 (step_cb event=token) ===
                from llm_client import get_client

                answer = ""
                _rc = ""
                _finished = False
                _emit_buf = ""

                async def _emit_token(tok):
                    """流式过滤: 抑制 <tool_calls>...</tool_calls> 块内 token, 其余实时推送。
                    (泄漏修复: 原实现先把 XML 实时发到前端, 流完才过滤 → 前端已显示污染)"""
                    nonlocal _emit_buf
                    import re as _re

                    _emit_buf += tok
                    while _emit_buf:
                        m = _re.search(r"<tool_calls>", _emit_buf)
                        if m is None:
                            out, _emit_buf = _emit_buf, ""
                        else:
                            out, _emit_buf = _emit_buf[: m.start()], _emit_buf[m.start() :]
                            end = _emit_buf.find("</tool_calls>")
                            if end == -1:
                                break  # 块未闭合 → 等后续 token
                            _emit_buf = _emit_buf[end + len("</tool_calls>") :]
                            continue
                        if out:
                            try:
                                await self._step_cb({"event": "token", "token": out})
                            except Exception:
                                pass

                try:
                    _stream = await get_client().chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=messages,
                        temperature=AGENT_TEMPERATURE,
                        max_tokens=AGENT_MAX_TOKENS,
                        stream=True,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    async for _ch in _stream:
                        if _ch.choices[0].finish_reason:
                            _finished = True
                        _delta = _ch.choices[0].delta.content or ""
                        if not _rc:
                            _rc = getattr(_ch.choices[0].delta, "reasoning_content", "") or ""
                        if _delta:
                            answer += _delta
                            await _emit_token(_delta)
                except Exception as e:
                    logger.error(f"LLM stream call failed at turn {turn_idx}: {e}")
                    final_result = f"LLM 调用失败: {e}"
                    break
                # 防截断续写 (流式末块未带 finish_reason → 续写, 不污染 messages)
                if not _finished and answer:
                    _cont = [{"role": "assistant", "content": answer}]
                    for _ in range(4):
                        _cont.append(
                            {"role": "user", "content": "（回答被截断，请从断点继续，不要重复已写内容）"}
                        )
                        _r2 = await chat_with_retry(
                            model=DEEPSEEK_MODEL,
                            messages=messages + _cont,
                            temperature=AGENT_TEMPERATURE,
                            max_tokens=AGENT_MAX_TOKENS,
                            extra_body={"thinking": {"type": "disabled"}},
                        )
                        _chunk = _r2.choices[0].message.content or ""
                        if _chunk:
                            answer += _chunk
                            await _emit_token(_chunk)
                        if _r2.choices[0].finish_reason != "length" or not _chunk:
                            break
                        _cont.append({"role": "assistant", "content": _chunk})
                # 过滤模型偶发把工具调用以 <tool_calls> XML 文本输出 (泄漏进答案)
                import re as _re

                answer = _re.sub(r"<tool_calls>.*?</tool_calls>", "", answer, flags=_re.S).strip()
                if _emit_buf:
                    # 收尾: 释放残留的块外文本 (流式缓冲未清空部分)
                    try:
                        await self._step_cb({"event": "token", "token": _emit_buf})
                    except Exception:
                        pass
                    _emit_buf = ""
                _am2 = {"role": "assistant", "content": answer}
                if _rc:
                    _am2["reasoning_content"] = _rc
                final_result = answer
                messages.append(_am2)

                # === HOOK: before_emit ===
                hr = self._fire_hook("before_emit", state, final_result)
                if hr.block:
                    final_result = hr.block_reason

                # DeepSeek 审查 #9: 直答轮也记录 turn, 使 trace.steps 完整
                turn.tool_success = True
                turn.elapsed_ms = (time.time() - t_turn_start) * 1000
                trace.turns.append(turn)

                # === Reflexion 反思重试闭环 (loop-engineering 第一环) ===
                # 直答后轻量自批判: 答案不完整/有遗漏 → 带批判意见重试一轮 (有界 1 次)
                if (
                    self.enable_reflexion
                    and not self._reflexed
                    and not hr.block
                    and turn_idx < self.max_steps - 1
                ):
                    critique = await self._reflexion_critique(messages)
                    if critique:
                        self._reflexed = True
                        messages.append(
                            {
                                "role": "user",
                                "content": f"【反思】{critique}\n请基于此批判修订你的回答，输出改进后的完整答案。",
                            }
                        )
                        continue  # 再跑一轮修订
                break

            turn.elapsed_ms = (time.time() - t_turn_start) * 1000
            trace.turns.append(turn)
            # 关键发现填充 (修复: 原 key_findings 从未写入 → 压缩摘要【关键发现】恒为"无"
            # → 压缩后证据丢失 → 幻觉风险。论文高精度场景必须保留跨压缩的证据。)
            try:
                if turn.tool_name and turn.tool_name != "llm":
                    prev = getattr(turn, "tool_result", "") or ""
                    if prev:
                        trace.key_findings.append(f"[{turn.tool_name}] {prev[:120]}")
                elif msg.content:
                    first_line = (msg.content or "").strip().split("\n")[0][:120]
                    if first_line:
                        trace.key_findings.append(f"[发现] {first_line}")
                trace.key_findings = trace.key_findings[-15:]  # 保留最近 15 条
            except Exception:
                pass
            # 步骤流事件 (Agent 流式: 每步完成即推送, 供 SSE 实时展示)
            if self._step_cb is not None:
                try:
                    await self._step_cb(
                        {
                            "event": "step",
                            "turn": turn.index + 1,
                            "tool": turn.tool_name or "llm",
                            "success": turn.tool_success,
                            "elapsed_ms": round(turn.elapsed_ms),
                            "tokens": turn.model_input_tokens
                            + turn.model_output_tokens,
                        }
                    )
                except Exception:
                    pass

        self._messages = messages
        self._state = state

        # === HOOK: trace_end ===
        self._fire_hook(
            "trace_end", {"result": final_result, "turns": len(trace.turns)}
        )

        return {
            "success": bool(final_result) and "超时" not in final_result,
            "result": final_result or "Agent 未能在限制步数内完成任务",
            # 统一 state 的消费点: 意图与实体随结果一起返回, 供前端展示/监控统计,
            # 而不是只写进 _state 就没人用
            "intent": self._state.get("intent") or {},
            "entities": self._state.get("entities") or {},
            "trace": {
                "goal": trace.goal,
                "turns": len(trace.turns),
                "tools_used": trace.tools_used,
                "total_tokens": trace.total_tokens,
                "elapsed": round(time.time() - trace.start_time, 1),
                "steps": [
                    {
                        "turn": t.index + 1,
                        "tool": t.tool_name or "llm",
                        "success": t.tool_success,
                        "tokens": t.model_input_tokens + t.model_output_tokens,
                        "elapsed_ms": round(t.elapsed_ms),
                    }
                    for t in trace.turns
                ],
            },
            "steps": state["steps"],
        }

    def _collect_remaining_confirms(self, msg, current_tc_id: str) -> list[dict]:
        """收集中间态: 当前 tc 之后**尚未应答**的兄弟 tool_call。

        为什么必须收: OpenAI 工具协议要求一条 assistant 消息里的每个 tool_call 都得有对应
        的 tool 响应, 否则下一次请求直接 400。此前遇到第一个待确认工具就 return, 其余兄弟
        既没执行也没应答 —— 恢复后消息列表是残缺的。
        """
        import json as _json

        out: list[dict] = []
        seen_current = False
        try:
            tool_calls = list(getattr(msg, "tool_calls", None) or [])
        except Exception:
            return out
        for tc in tool_calls:
            if not seen_current:
                if tc.id == current_tc_id:
                    seen_current = True
                continue
            name = tc.function.name
            try:
                args = _json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            out.append(
                {
                    "tool_name": name,
                    "tool_args": args,
                    "tool_call_id": tc.id,
                }
            )
        return out

    def _persist_pending(self) -> None:
        """把中断快照落盘 —— 进程重启后仍能恢复 (否则用户点了确认却被告知无待确认)。"""
        try:
            from database import get_db

            pending = []
            if self._pending_confirm:
                pending.append(self._pending_confirm)
            pending.extend(self._pending_queue)
            get_db().save_pending_run(
                run_id=self.run_id,
                user_id=getattr(self, "owner", "") or "",
                goal=self.goal or "",
                messages=self._messages,
                state=self._state,
                pending=pending,
                meta={
                    "max_steps": self.max_steps,
                    "skills": getattr(self, "_active_skills", []) or [],
                },
            )
        except Exception as e:
            logger.warning(f"persist pending run failed: {e}")

    def _clear_pending(self) -> None:
        try:
            from database import get_db

            get_db().clear_pending_run(self.run_id)
        except Exception as e:
            logger.warning(f"clear pending run failed: {e}")

    async def _drain_pending_queue(self) -> Optional[dict]:
        """顺序处理排队中的兄弟 tool_call。

        返回: 下一个待确认项 (需要再次暂停) 或 None (队列已清空, 可以继续 _loop)。
        不需确认的直接执行并补上 tool 响应, 保证消息列表始终完整。
        """
        from agent.tool_registry import get_tool

        while self._pending_queue:
            nxt = self._pending_queue.pop(0)
            tool_def = get_tool(nxt["tool_name"])
            if tool_def is not None and tool_def.need_confirm:
                self._pending_confirm = nxt
                self._persist_pending()
                return {
                    "success": True,
                    "paused": True,
                    "run_id": self.run_id,
                    "need_confirm": self._pending_confirm,
                    "queued": len(self._pending_queue),
                    "result": f"需要人工确认执行工具: {nxt['tool_name']}",
                }
            # 无需确认 → 直接执行, 补上响应
            if tool_def is None:
                content = f"工具 {nxt['tool_name']} 不存在"
            else:
                _ok, _res = await self.tool_mgr.execute_with_retry(
                    nxt["tool_name"], tool_def, nxt["tool_args"]
                )
                content = (
                    str(_res)[:3000]
                    if _ok
                    else self.tool_mgr.format_error_for_llm(
                        nxt["tool_name"], str(_res), nxt["tool_args"]
                    )
                )
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": nxt["tool_call_id"],
                    "content": content,
                }
            )
        self._pending_confirm = None
        self._clear_pending()
        return None

    async def resume(self, confirm: bool = True) -> dict:
        """人工确认后恢复执行 (合并自 langgraph resume_langgraph_agent)。

        confirm=True → 执行被暂停的危险工具; False → 拒绝并让 Agent 换路。
        """
        if not self._pending_confirm:
            return {
                "success": False,
                "result": "没有待确认的工具调用",
                "run_id": self.run_id,
            }
        pc = self._pending_confirm
        # === Plan 模式: 计划确认 (与工具确认共用本入口, 用 type 区分) ===
        if pc.get("type") == "plan":
            self._pending_confirm = None
            self._clear_pending()
            if not confirm:
                self._fire_hook("trace_end", {"result": "plan_rejected"})
                return {
                    "success": False,
                    "result": "计划已取消。请重新描述任务，我会给出新计划。",
                    "run_id": self.run_id,
                }
            # 确认 → 计划已在 _messages 中 (作为 assistant 消息), 直接续跑
            self._fire_hook("trace_start", {"task": "plan_approved"})
            result = await self._loop()
            # 收尾: 与 run() 一致的持久化 (成功策略 + 课题记忆)
            _task = ""
            for _m in self._messages:
                if _m.get("role") == "user":
                    _task = str(_m.get("content", ""))[:80]
                    break
            try:
                if result.get("success"):
                    from agent_skills import record_success

                    record_success(
                        _task,
                        (result.get("trace") or {}).get("tools_used") or [],
                        str(result.get("result", ""))[:150],
                        user_id=self.owner,
                    )
            except Exception:
                pass
            if self.topic_id:
                try:
                    from conversation_memory import get_memory

                    _mem = get_memory(self.topic_id)
                    _tr = result.get("trace") or {}
                    for _t in _tr.get("tools_used") or []:
                        _mem.add_tool_call(_t, "")
                    for _kf in _tr.get("key_findings") or []:
                        _mem.add_finding(str(_kf)[:300], "agent")
                except Exception:
                    pass
            return result

        from agent.tool_registry import get_tool

        tool_def = get_tool(pc["tool_name"])
        if confirm and tool_def is not None:
            _ok, _res = await self.tool_mgr.execute_with_retry(
                pc["tool_name"], tool_def, pc["tool_args"]
            )
            content = (
                str(_res)[:3000]
                if _ok
                else self.tool_mgr.format_error_for_llm(
                    pc["tool_name"], str(_res), pc["tool_args"]
                )
            )
        elif confirm:
            content = f"工具 {pc['tool_name']} 不存在, 无法执行"
        else:
            content = "用户拒绝了此工具调用, 请改用其他方法完成任务。"

        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": pc["tool_call_id"],
                "content": content,
            }
        )
        self._pending_confirm = None
        # 还有排队的兄弟 tool_call → 依次处理; 若其中有待确认的则再次暂停
        paused = await self._drain_pending_queue()
        if paused:
            return paused
        return await self._loop()

    def _build_system_prompt(self, skill_prompt: str = "", task: str = "") -> str:
        """构建系统提示: 基础角色 + 激活技能附录 + 反馈教训(P2) + 相似成功策略(P3) + 通用约束。"""
        cap = f"\n\n{skill_prompt}" if skill_prompt else ""
        # P2 反馈驱动调整: 注入低分反馈提炼的「易失败教训」
        lessons_txt = ""
        try:
            from feedback_lessons import get_failure_lessons

            lessons = get_failure_lessons(user_id=self.owner)
            if lessons:
                lessons_txt = "\n\n【历史反馈教训（用户对下列问题类型给过低分，务必避免同类错误）】\n" + "\n".join(
                    f"- {l}" for l in lessons
                )
        except Exception:
            pass
        # 跨 agent 记忆: 注入课题累积的关键发现 (此前交互 + 科研团队产出)。
        # 修复前: pi_agent 只**写**课题记忆不**读** → research_team 的产出 pi_agent 看不到,
        # 跨 agent 记忆只单向 (只有问答路径 ask_question 读)。
        mem_txt = ""
        if self.topic_id:
            try:
                from conversation_memory import get_memory

                _mctx = get_memory(self.topic_id).get_context(task)
                if _mctx:
                    mem_txt = f"\n\n【课题记忆（此前交互/科研团队产出的关键发现，可复用）】\n{_mctx[:1500]}"
            except Exception:
                pass
        # P3 持久技能记忆: 注入相似成功任务的已验证做法
        strat_txt = ""
        try:
            from agent_skills import get_similar_strategies

            for s in get_similar_strategies(self.goal or task or "", user_id=self.owner):
                strat_txt += (
                    f"\n- 相似任务「{str(s.get('task',''))[:40]}」成功做法: "
                    f"调用工具 [{s.get('tools_used','')}]"
                )
            if strat_txt:
                strat_txt = "\n\n【过往相似任务的成功经验（可参考此做法）】" + strat_txt
        except Exception:
            pass
        kb_txt = f"\n\n【当前课题知识库作用域】\n{self.kb_context}" if self.kb_context else ""
        return f"""你是 PaperLink 学术智能助手。{cap}{kb_txt}{mem_txt}{lessons_txt}{strat_txt}

---
工作流: 理解任务 → 优先 search_knowledge_bases/当前课题库 → 问答/agent_ask → 综合分析 → 输出报告; 只有库内无相关内容才外部搜索
约束: 每个回复最多调用一个工具 | 引用使用 [论文: 标题, 年份] 格式
纯净输出: 只输出最终结果/回答; 严禁在回复中出现「让我查询/我需要补充/我再确认/让我再」等过程性文字, 严禁输出任何内部思考、工具调用内容或 XML 标记
学术规范: 引用必须真实 (库内文献用 [编号], 非库内共识明确标注「通用学术共识」), 不编造数据; 结论诚实标注局限
目标: {self.goal or '高效完成学术调研任务'}"""


# ============================================================
# Agent 实例注册表 — 按 run_id 隔离, 支持人工确认暂停后跨请求 resume
# ============================================================
_pi_agents: dict[str, PiAgent] = {}
_pi_agents_lock = threading.Lock()


def _describe_intent(task: str, selected_skills: list[str]) -> dict:
    """统一 state 的意图槽: 技能路由结果 + 规则/语义意图分类。

    复用已有的三层漏斗 (规则 → 嵌入语义 → 默认), 不额外调 LLM。
    """
    intent: dict = {"skills": selected_skills, "label": "", "source": ""}
    try:
        from qa_engine import _classify_intent

        intent["label"] = _classify_intent(task)
        intent["source"] = "rules+semantic"
    except Exception:
        pass
    return intent


def _extract_entities(task: str) -> dict:
    """从任务文本抽取结构化实体 (纯规则, 零 LLM 调用)。

    学术场景真正有用的槽位: arXiv id / DOI / 年份 / 引号里的术语 / 显式列出的关键词。
    抽不到就留空, 不猜 —— 猜出来的实体比没有更糟 (下游会当成事实用)。
    """
    import re as _re

    t = task or ""
    ents: dict = {}
    dois = _re.findall(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", t)
    if dois:
        ents["dois"] = dois
    # arXiv id 与 DOI 的形态会撞车 (DOI `10.1109/CVPR.2019.00677` 里的 `2019.00677`
    # 也匹配 YYMM.NNNNN)。先剔除 DOI 片段再匹配, 否则会凭空造出一个假的 arXiv id。
    _t_no_doi = t
    for d in dois:
        _t_no_doi = _t_no_doi.replace(d, " ")
    arxiv = _re.findall(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?![\d.])", _t_no_doi)
    if arxiv:
        ents["arxiv_ids"] = [a[0] for a in arxiv]
    years = sorted({int(y) for y in _re.findall(r"\b((?:19|20)\d{2})\b", t)})
    if years:
        ents["years"] = years
    quoted = _re.findall(r"[「『\"“]([^」』\"”]{2,40})[」』\"”]", t)
    if quoted:
        ents["quoted_terms"] = quoted[:8]
    return ents


def _restore_agent(run_id: str) -> Optional["PiAgent"]:
    """从 SQLite 中断快照重建被暂停的 agent (进程重启后 resume 仍可用)。

    只重建恢复所需的最小状态 (messages / state / 待确认队列), 不重跑任何 LLM 调用 ——
    用户先前烧掉的 token 不会白费。
    """
    try:
        from database import get_db

        snap = get_db().get_pending_run(run_id)
    except Exception as e:
        logger.warning(f"load pending snapshot failed: {e}")
        return None
    if not snap:
        return None

    meta = snap.get("meta") or {}
    agent = PiAgent(
        goal=snap.get("goal") or "",
        max_steps=int(meta.get("max_steps") or 15),
    )
    agent.run_id = run_id  # 沿用原 run_id, 前端持有的那个才有意义
    agent.owner = snap.get("user_id") or ""
    agent._messages = snap.get("messages") or []
    agent._state = snap.get("state") or {}
    pending = snap.get("pending") or []
    agent._pending_confirm = pending[0] if pending else None
    agent._pending_queue = pending[1:]
    if agent._pending_confirm is None:
        return None  # 快照里没有待确认项 → 不是可恢复的中断态
    return agent


def get_agent(run_id: str = "", **kwargs) -> PiAgent:
    """按 run_id 获取/注册 PiAgent。

    - run_id 为空: 创建全新实例 (每个 /agent/pi 独立 run)。
    - run_id 在内存: 直接返回 (resume 续跑)。
    - run_id 不在内存但库里有中断快照: **从快照重建** (进程重启后仍可续跑)。
    - 都没有: 抛 KeyError (由 api 层转 404)。
    """
    if run_id:
        with _pi_agents_lock:
            agent = _pi_agents.get(run_id)
            if agent is not None:
                return agent
        restored = _restore_agent(run_id)
        if restored is not None:
            with _pi_agents_lock:
                _pi_agents[run_id] = restored
            logger.info(f"Restored paused agent from snapshot: {run_id}")
            return restored
        raise KeyError(f"run_id 不存在: {run_id}")

    agent = PiAgent(**kwargs)
    with _pi_agents_lock:
        _pi_agents[agent.run_id] = agent
        _prune_agents()
    return agent


def _prune_agents() -> None:
    """注册表上限 200。超限先释放无待确认的 run, 再丢最旧。"""
    if len(_pi_agents) < 200:
        return
    for rid in [r for r, a in _pi_agents.items() if a._pending_confirm is None]:
        _pi_agents.pop(rid, None)
    while len(_pi_agents) > 200:
        _pi_agents.pop(next(iter(_pi_agents)), None)
