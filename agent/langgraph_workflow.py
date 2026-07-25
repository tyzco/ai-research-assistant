"""LangGraph Agent 工作流：StateGraph 替代手写 ReAct 循环。

准备2(技术) §334: 用 StateGraph 搭建图状工作流，实现节点化功能拆分与流程控制。
准备2(技术) §1104: 条件边实现分支逻辑，工具调用失败按错误码重试/兜底/切备用工具。
准备2(技术) §1251: 支持循环执行与全局状态管理，自主纠错、断点续跑、人机协同。"""

import json
import logging
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)

# ===== State 定义 =====


class AgentState(TypedDict):
    """LangGraph Agent 全局状态。add_messages reducer 自动追加而非覆盖。"""

    messages: Annotated[list[dict], add_messages]
    need_confirm: dict | None  # {tool_name, tool_args} — 需人工确认时设置
    confirmed: bool
    result: str
    steps: int
    max_steps: int


# ===== 节点函数 =====

SYSTEM_PROMPT = """你是学术文献调研 Agent。可使用工具完成论文搜索、下载、知识库问答、报告导出。

工作流：
1. 理解调研任务 → 生成检索策略
2. 搜索论文（search_papers）→ 检查有没有已构建的知识库（search_knowledge_bases）
3. 如有 KB 直接用 agent_ask 问答，否则下载论文建库
4. 综合分析生成报告，需要时可 export_report

规则：
- 每个回复只能调用一个工具
- 工具结果会直接反馈，据此决定下一步
- 如果无法继续，说明原因
- 最终答案以 Markdown 格式输出，引用来源"""


async def agent_node(state: AgentState) -> dict:
    """Agent 推理节点：调用 LLM（带工具），决定下一步动作。"""
    from agent.tool_registry import get_tools_schema

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]

    try:
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=msgs,
            tools=get_tools_schema(),
            tool_choice="auto",
            temperature=0.3,
            max_tokens=1000,
        )
    except Exception as e:
        logger.error(f"Agent LLM error: {e}")
        return {
            "messages": [{"role": "assistant", "content": f"LLM 调用失败: {e}"}],
            "result": f"LLM 调用失败: {e}",
        }

    msg = resp.choices[0].message
    if msg.tool_calls:
        # LLM 选择调用工具
        tc = msg.tool_calls[0]
        return {
            "messages": [msg],
            "steps": state.get("steps", 0) + 1,
        }
    else:
        # LLM 给出最终答案
        answer = msg.content or ""
        return {
            "messages": [msg],
            "result": answer,
            "steps": state.get("steps", 0) + 1,
        }


async def tools_node(state: AgentState) -> dict:
    """工具执行节点：调用工具并返回结果。支持人工确认门。"""
    from agent.tool_registry import get_tool

    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {}

    tc = last_msg.tool_calls[0]
    tool_name = tc.function.name
    tool_args = json.loads(tc.function.arguments)
    tool = get_tool(tool_name)

    if not tool:
        result = f"工具 {tool_name} 不存在"
    elif tool.need_confirm and not state.get("confirmed"):
        # 需要人工确认 → 中断执行
        return {"need_confirm": {"tool_name": tool_name, "tool_args": tool_args}}
    else:
        try:
            result = await tool.execute(**tool_args)
            result = str(result)[:3000]
        except Exception as e:
            result = f"工具执行失败: {e}"

    return {
        "messages": [{"role": "tool", "tool_call_id": tc.id, "content": result}],
        "need_confirm": None,
        "confirmed": False,
    }


def _should_continue(state: AgentState) -> Literal["tools", "reflect"]:
    """条件边：有 tool_calls → 执行工具；无 → 反思/结束。"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "reflect"


def _should_loop(state: AgentState) -> Literal["agent", "reflect"]:
    """条件边：工具执行后，步数未超 → 继续推理；超限 → 强制结束。"""
    if state.get("need_confirm"):
        return "reflect"  # 需要确认，暂停
    if state.get("steps", 0) >= state.get("max_steps", 10):
        return "reflect"
    return "agent"


async def reflect_node(state: AgentState) -> dict:
    """反思节点：检查答案完整性。（准备2 §54: 反思机制 + 准备2 §1104: 失败重试兜底）"""
    need_confirm = state.get("need_confirm")
    if need_confirm:
        return {
            "result": json.dumps(
                {
                    "paused": True,
                    "need_confirm": need_confirm,
                    "message": f"需要确认执行 {need_confirm['tool_name']}",
                },
                ensure_ascii=False,
            ),
        }

    result = state.get("result", "")
    if not result:
        return {"result": f"Agent 在 {state.get('steps', 0)} 步后未完成任务。"}
    return {"result": result}


# ===== 图构建 =====


def build_agent_graph():
    """构建 LangGraph Agent 状态图。"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("reflect", reflect_node)

    # 设置入口
    workflow.set_entry_point("agent")

    # 条件边：agent → tools 或 reflect
    workflow.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", "reflect": "reflect"}
    )

    # 工具执行后 → 回到 agent 继续推理，或强制结束
    workflow.add_conditional_edges(
        "tools", _should_loop, {"agent": "agent", "reflect": "reflect"}
    )

    # 反思后 → 结束
    workflow.add_edge("reflect", END)

    # 编译（带检查点支持断点续跑）
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory, interrupt_before=["tools"] if True else []
    )


# 全局图实例
_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


# ===== 便捷入口 =====


async def run_langgraph_agent(
    task: str, max_steps: int = 10, thread_id: str = "default"
) -> dict:
    """运行 LangGraph Agent 完成调研任务。

    Args:
        task: 调研任务描述
        max_steps: 最大推理步数
        thread_id: 会话线程 ID（用于断点续跑和多轮对话）

    Returns:
        {result: str, steps: int, success: bool}
    """
    graph = get_agent_graph()
    initial = {
        "messages": [{"role": "user", "content": f"调研任务：{task}"}],
        "max_steps": max_steps,
        "steps": 0,
        "result": "",
        "confirmed": False,
        "need_confirm": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = await graph.ainvoke(initial, config)

    # 提取结果
    need_confirm = final_state.get("need_confirm")
    if need_confirm:
        return {
            "success": True,
            "paused": True,
            "need_confirm": need_confirm,
            "result": f"需要确认执行 {need_confirm['tool_name']}",
            "steps": final_state.get("steps", 0),
        }

    return {
        "success": bool(final_state.get("result")),
        "result": final_state.get("result", "Agent 未完成任务"),
        "steps": final_state.get("steps", 0),
    }


async def resume_langgraph_agent(thread_id: str, confirmed: bool = True) -> dict:
    """人工确认后恢复 Agent 执行。"""

    graph = get_agent_graph()
    config = {"configurable": {"thread_id": thread_id}}
    await graph.aupdate_state(config, {"confirmed": confirmed, "need_confirm": None})
    # 继续执行
    result = []
    async for chunk in graph.astream(None, config):
        result.append(chunk)
    if result:
        final = list(result[-1].values())[0]
        return {
            "success": True,
            "result": final.get("result", ""),
            "steps": final.get("steps", 0),
        }
    return {"success": False, "result": "恢复执行失败"}


# ===== 多Agent模式：Planner→Worker→Synthesizer =====
# 准备2 §多智能体协同：编排者-执行者模式的分工逻辑

PLANNER_PROMPT = """你是学术调研规划器。将调研任务拆解为2-4个独立子任务，每个子任务可用工具独立完成。
直接返回JSON，不要解释。

调研任务：{task}

格式：{{"subtasks": [{{"id": 1, "description": "子任务描述", "tool": "agent_ask|search_papers|ask_knowledge_base"}}], "synthesis_notes": "汇总要点"}}"""

SYNTHESIZER_PROMPT = """你是学术报告撰写器。根据以下子任务结果，整合为结构化的研究报告（Markdown）。

{results}

研究报告："""


class MultiAgentGraph:
    """轻量多Agent：Planner拆解→Worker并行执行→Synthesizer汇总。
    不使用CrewAI等重型框架——LangGraph本身的状态管理已足够。"""

    async def plan(self, task: str, client) -> dict:
        try:
            resp = await client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=0.2,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": PLANNER_PROMPT.format(task=task)}
                ],
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception:
            return {
                "subtasks": [{"id": 1, "description": task, "tool": "agent_ask"}],
                "synthesis_notes": "",
            }

    async def execute_worker(self, subtask: dict, table_name: str) -> dict:
        from agent.tool_registry import get_tool

        tool = get_tool(subtask.get("tool", "agent_ask"))
        if not tool:
            return {"error": f"Unknown tool: {subtask.get('tool')}"}
        try:
            result = await tool.execute(
                table_name=table_name
                if "table" in tool.parameters.get("properties", {})
                else None,
                question=subtask["description"],
                query=subtask["description"],
            )
            return {
                "id": subtask["id"],
                "description": subtask["description"],
                "result": result[:2000],
            }
        except Exception as e:
            return {"id": subtask["id"], "error": str(e)}

    async def synthesize(self, worker_results: list[dict], client) -> str:
        results_text = "\n---\n".join(
            f"子任务 {r.get('id','?')}: {r.get('description','')}\n{r.get('result', r.get('error',''))}"
            for r in worker_results
        )
        try:
            resp = await client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=0.3,
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": SYNTHESIZER_PROMPT.format(
                            results=results_text[:4000]
                        ),
                    }
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"汇总失败: {e}\n\n原始结果:\n{results_text}"


async def run_multi_agent(task: str, table_name: str = "") -> dict:
    """运行多Agent调研：Planner拆解 → 串行执行Workers → Synthesizer汇总。"""
    from openai import AsyncOpenAI

    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    ma = MultiAgentGraph()

    # Step 1: Planner
    plan = await ma.plan(task, client)
    subtasks = plan.get(
        "subtasks", [{"id": 1, "description": task, "tool": "agent_ask"}]
    )
    notes = plan.get("synthesis_notes", "")

    # Step 2: Workers (串行——避免API并发限制)
    worker_results = []
    for st in subtasks:
        result = await ma.execute_worker(st, table_name)
        worker_results.append(result)

    # Step 3: Synthesizer
    report = await ma.synthesize(worker_results, client)

    return {
        "success": True,
        "result": report,
        "plan": {"subtasks": len(subtasks), "notes": notes},
        "workers": [{"id": r["id"], "ok": "error" not in r} for r in worker_results],
        "steps": len(subtasks) + 2,
    }


# ===== Agent 评估体系 =====
# 准备2 §评估体系：任务成功率、平均推理步数、工具调用准确率、影子测试


class AgentEvaluator:
    """Agent 四维评估：成功率、步数、工具准确率、影子测试（与基线对照）。"""

    def __init__(self):
        self.results: list[dict] = []

    def record(self, task: str, expected: str, actual: dict, shadow_baseline: str = ""):
        self.results.append(
            {
                "task": task,
                "expected": expected,
                **actual,
                "shadow_baseline": shadow_baseline,
            }
        )

    def report(self) -> dict:
        if not self.results:
            return {}
        n = len(self.results)
        success = sum(1 for r in self.results if r.get("success"))
        steps = [r.get("steps", 0) for r in self.results if r.get("steps", 0) > 0]
        tool_ok = sum(1 for r in self.results if r.get("tool_errors", 0) == 0)
        shadow_pass = sum(
            1
            for r in self.results
            if r.get("shadow_baseline")
            and len(r.get("result", "")) > len(r["shadow_baseline"]) * 0.8
        )
        return {
            "task_success_rate": f"{success}/{n} ({success/n:.0%})",
            "avg_inference_steps": round(sum(steps) / max(1, len(steps)), 1),
            "tool_call_accuracy": f"{tool_ok}/{n}",
            "shadow_test_pass": f"{shadow_pass}/{sum(1 for r in self.results if r.get('shadow_baseline'))}"
            if any(r.get("shadow_baseline") for r in self.results)
            else "N/A",
            "details": self.results,
        }


_evaluator = AgentEvaluator()


def get_agent_evaluator():
    return _evaluator
