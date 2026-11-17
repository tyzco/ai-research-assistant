"""可视化运行记录与图数据路由 (/flow/*)。"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request

from lance_utils import list_lance_tables as _list_lance_tables
from monitor import sanitize_input

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/flow/runs")
async def api_flow_save_run(request: Request):
    """保存一次可视化运行 (任务 + 参数快照 + 完整事件 + 结果), 供历史回放。"""
    import json as _json

    req = await request.json()
    events = req.get("events") or []
    if not isinstance(events, list):
        raise HTTPException(400, "events 必须是数组")
    # 防止超大 payload 占满 SQLite: 单次运行事件通常 < 1MB
    if len(_json.dumps(events, ensure_ascii=False)) > 5 * 1024 * 1024:
        raise HTTPException(413, "events 过大")
    run_id = sanitize_input(req.get("run_id", "")) or f"flow_{uuid.uuid4().hex[:12]}"
    from database import get_db

    get_db().save_flow_run(
        run_id=run_id,
        task=sanitize_input(req.get("task", "")),
        params=req.get("params") or {},
        events=events,
        result=str(req.get("result", ""))[:20000],
        status=sanitize_input(req.get("status", "")),
        total_ms=float(req.get("total_ms", 0) or 0),
        agent_count=int(req.get("agent_count", 0) or 0),
        batch_count=int(req.get("batch_count", 0) or 0),
        trace_id=sanitize_input(req.get("trace_id", "")),
    )
    return {"ok": True, "run_id": run_id}


@router.get("/flow/runs")
async def api_flow_list_runs(limit: int = 10):
    """最近的可视化运行 (不含 events)。"""
    from database import get_db

    return {"runs": get_db().list_flow_runs(limit=min(max(limit, 1), 50))}


@router.get("/flow/runs/{run_id}")
async def api_flow_get_run(run_id: str):
    """单次运行的完整记录 (params / events 已解析)。"""
    from database import get_db

    d = get_db().get_flow_run(run_id)
    if not d:
        raise HTTPException(404, f"运行不存在: {run_id}")
    return d


@router.get("/flow/kbs")
async def api_flow_kbs():
    """RAG 模式可选的知识库表 (按行数排序)。"""
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    out = []
    for t in _list_lance_tables(db):
        try:
            out.append({"table": t, "rows": db.open_table(t).count_rows()})
        except Exception:
            out.append({"table": t, "rows": 0})
    out.sort(key=lambda x: -x["rows"])
    return {"kbs": out}


def _flow_metrics(run: dict) -> dict:
    """从事件流提炼对比指标: 总耗时、token、每个 Agent 的耗时、结果长度。"""
    stage_start: dict = {}
    agent_ms: dict = {}
    tokens = 0
    rag_stages: list = []
    rag_ms: dict = {}
    _last_stage = None
    _last_t = None
    for e in run.get("events") or []:
        ev = e.get("ev") or {}
        t = float(e.get("t") or 0)
        if ev.get("event") == "stage":
            k = ev.get("agent")
            if ev.get("status") == "running":
                stage_start[k] = t
            elif ev.get("status") == "done" and k in stage_start:
                agent_ms[k] = round(t - stage_start[k], 1)
        elif ev.get("event") == "agent_llm":
            tokens += int(ev.get("tokens") or 0)
        elif ev.get("event") == "rag":
            _stage = ev.get("stage")
            if _last_stage and _last_t is not None:
                rag_ms[_last_stage] = rag_ms.get(_last_stage, 0) + (t - _last_t)
            _last_stage = _stage
            _last_t = t
            rag_stages.append(_stage)
    return {
        "run_id": run.get("run_id"),
        "task": run.get("task", ""),
        "params": run.get("params") or {},
        "total_ms": round(float(run.get("total_ms") or 0), 1),
        "tokens": tokens,
        "agent_ms": agent_ms,
        "rag_ms": {k: round(v, 1) for k, v in rag_ms.items()},
        "rag_calls": len(rag_stages),
        "result_len": len(run.get("result") or ""),
        "events": len(run.get("events") or []),
        "created_at": run.get("created_at", ""),
    }


@router.post("/flow/compare")
async def api_flow_compare(request: Request):
    """对比两次运行: 指标差异 + LLM 生成的分析报告。"""
    from database import get_db

    req = await request.json()
    a_id = sanitize_input(req.get("a", ""))
    b_id = sanitize_input(req.get("b", ""))
    if not a_id or not b_id:
        raise HTTPException(400, "需要 a 和 b 两个 run_id")
    a = get_db().get_flow_run(a_id)
    b = get_db().get_flow_run(b_id)
    if not a or not b:
        raise HTTPException(404, "运行记录不存在")
    ma, mb = _flow_metrics(a), _flow_metrics(b)

    report = ""
    try:
        from llm_client import chat_with_retry

        from config import DEEPSEEK_MODEL

        def _brief(m: dict) -> str:
            return (
                f"任务: {m['task']}\n"
                f"参数: {m['params']}\n"
                f"总耗时: {m['total_ms']}ms · token: {m['tokens']} · "
                f"结果字数: {m['result_len']} · 事件数: {m['events']}\n"
                f"各 Agent 耗时(ms): {m['agent_ms']}\n"
                f"RAG 阶段调用次数: {m['rag_calls']}"
            )

        resp = await chat_with_retry(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "你是 Agent 工作流的性能分析员。对比下面两次运行, 输出 Markdown:"
                        "1) 参数差异及其对应影响; 2) 耗时/token 差异与可能原因;"
                        "3) 哪个环节变化最大; 4) 3 条可执行的调优建议。"
                        "只依据给定数据, 不确定就说不确定, 不要编造。\n\n"
                        f"【运行 A】\n{_brief(ma)}\n\n【运行 B】\n{_brief(mb)}"
                    ),
                }
            ],
            temperature=0.2,
            max_tokens=1200,
            extra_body={"thinking": {"type": "disabled"}},
            retries=1,
        )
        report = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"flow compare report failed: {e}")
        report = "（LLM 分析不可用，下面是原始指标对比）"

    return {"a": ma, "b": mb, "report": report}


@router.get("/flow/graph")
async def api_flow_graph():
    """Agent 工作流可视化数据源 (`static/flow.html` 消费)。

    设计原则: agents / tools / hooks / skills **直接从代码导入**, 不维护手写副本。
    手写一份流程图最容易出现的失败, 就是图和代码各改各的、最后互相对不上 ——
    那等于把黑箱从代码搬到了图上。这里只有执行顺序的阶段描述来自
    `agent/flow_meta.py` (代码本身不自描述顺序), 且每段都带代码锚点。
    """
    from agent.flow_meta import FALLBACKS, GATES, LAYERS, PI_LOOP
    from agent.hooks import HookPoint, get_hooks
    from agent.research_team import AGENT_DEFINITIONS
    from agent.skills import get_skills
    from agent.tool_registry import TOOLS

    gate_names = {g["id"]: g["name"] for g in GATES}
    agents = []
    for key, d in AGENT_DEFINITIONS.items():
        g = d.get("needs_approval")
        gv = getattr(g, "value", None)
        agents.append(
            {
                "key": key,
                "name": d.get("name", key),
                "layer": d.get("layer", ""),
                "role": (d.get("role") or "")[:300],
                "tools": d.get("tools", []),
                "gate": gv,
                "gate_name": gate_names.get(gv, ""),
            }
        )

    tools = [
        {
            "name": t.name,
            "description": (t.description or "")[:200],
            "need_confirm": bool(t.need_confirm),
            "kind": t.kind,
            "retryable": bool(t.retryable),
            "cacheable": bool(t.cacheable),
            "requires_precheck": t.requires_precheck or "",
        }
        for t in TOOLS
    ]

    # 可视化参数面板的"代码默认值" —— 前端展示"为什么是这个值"必须来自真实代码
    from config import (
        AGENT_AGENT_TIMEOUT,
        AGENT_MAX_TOKENS,
        AGENT_TEMPERATURE,
        DEEPSEEK_MODEL,
        RERANK_MIN_SCORE,
        TEAM_AGENT_MAX_ROUNDS,
    )

    defaults = {
        "model": DEEPSEEK_MODEL,
        "temperature": AGENT_TEMPERATURE,
        "max_tokens": AGENT_MAX_TOKENS,
        "agent_max_rounds": TEAM_AGENT_MAX_ROUNDS,
        "agent_timeout": AGENT_AGENT_TIMEOUT,
        "simple_max_len": 60,
        "rag_top_k": 5,
        "rerank_min_score": RERANK_MIN_SCORE,
        "parallel_enabled": True,
    }

    registry = get_hooks()
    hook_points = [p.value for p in HookPoint]
    registered = registry.list_registered()
    return {
        "defaults": defaults,
        "layers": [
            {**layer, "agents": [a["key"] for a in agents if a["layer"] == layer["id"]]}
            for layer in LAYERS
        ],
        "agents": agents,
        "gates": GATES,
        "pi_loop": PI_LOOP,
        "fallbacks": FALLBACKS,
        "tools": tools,
        "hooks": {
            "points": hook_points,
            "registered": registered,
            "stats": registry.get_stats(),
        },
        "skills": get_skills().list_all(),
        "counts": {
            "agents": len(agents),
            "layers": len(LAYERS),
            "gates": len(GATES),
            "tools": len(tools),
            "hook_points": len(hook_points),
            "registered_hooks": sum(len(v) for v in registered.values()),
            "skills": len(get_skills().list_all()),
        },
    }


