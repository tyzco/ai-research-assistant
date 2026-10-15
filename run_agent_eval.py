"""Agent 级评测 — 多步规划 / 工具调用 / 兜底触发 的量化指标。

与另外两级评测互补:
  run_eval.py                 → 检索层 (R@5 / MRR)
  evaluate_rag_standards.py   → 生成层 (faithfulness / relevancy / recall / precision)
  本脚本                       → **Agent 层** (规划与工具调用)

为什么单独做一级: 前两级都只看"检索到没/答得准不准", 看不到"这个任务是不是被拆对了、
工具是不是被用对了、有没有反复空转"。而这些恰恰是 agent 项目最容易退化的地方 ——
改了系统提示、改了工具集、改了调度, 检索指标可能纹丝不动, 但 agent 从"3 步做完"
变成"8 步还在试"。

指标 (全部可从 trace 直接观测, 不做主观打分):
  完成率        success 比例
  平均步数      每任务消耗的 turn 数
  工具调用数     实际执行的工具次数 (不含被拦下的)
  无效调用占比   失败 / 重复 / 被兜底拦下的工具调用比例 —— 衡量"有没有空转"
  兜底触发     降级计数器 (repeat_action / param_budget / error_accumulation 等) 的增量
  平均耗时 / tokens

用法:
    python run_agent_eval.py              # 跑全部内置任务
    python run_agent_eval.py --limit 1    # 只跑第一个 (省成本, 冒烟用)
    python run_agent_eval.py --baseline   # 与上次结果对比 (回归门禁)

前提: .env 已配置 DEEPSEEK_API_KEY; 会创建临时课题并在结束时清理。
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("agent_eval")

REPORT_PATH = Path("data/agent_eval_result.json")

# 内置任务集: 刻意覆盖"单步 / 多步 / 需要工具 / 不需要工具 / 应当坦白做不到"几类。
# 全用真实 LLM 跑 —— 假 LLM 只能验证 harness, 测不出规划能力。
TASKS = [
    {
        "id": "single_step_state",
        "task": "查看当前课题的状态：有多少篇论文、知识库是否就绪。",
        "expect_success": True,
        "note": "单步: 应调用 get_topic_state 一次就收",
    },
    {
        "id": "kb_question",
        "task": "基于当前课题的知识库回答：这篇论文的核心结论是什么？",
        "expect_success": True,
        "note": "多步: 需先确认库存在, 再 agent_ask",
    },
    {
        "id": "should_not_hallucinate",
        "task": "请直接告诉我量子色动力学中胶子传播子的精确表达式。",
        "expect_success": True,
        "note": "库内必无此内容 —— 看它会不会硬编 (衡量幻觉倾向)",
    },
]


def _snapshot_degradations() -> dict:
    try:
        from monitor import metrics

        return dict(metrics.degradation_stats() or {})
    except Exception:
        return {}


def _diff_degradations(before: dict, after: dict) -> dict:
    out = {}
    for k, v in (after or {}).items():
        d = (v or 0) - (before.get(k) or 0)
        if d:
            out[k] = d
    return out


async def run_one(task_def: dict, topic_id: str, table_name: str) -> dict:
    """跑一个任务, 返回可从 trace 直接观测的指标。"""
    from agent.pi_agent import PiAgent

    before = _snapshot_degradations()
    t0 = time.time()
    rec = {
        "id": task_def["id"],
        "note": task_def.get("note", ""),
        "success": False,
        "turns": 0,
        "tool_calls": 0,
        "elapsed": 0.0,
        "tokens": 0,
        "steps_failed": 0,
        "tools": [],
        "degradations": {},
        "answer_preview": "",
        "error": "",
    }
    try:
        agent = PiAgent(
            goal=task_def["task"][:80],
            topic_id=topic_id,
            kb_context=f"当前课题知识库表名: {table_name}",
        )
        res = await agent.run(task_def["task"])
        trace = res.get("trace") or {}
        steps = trace.get("steps") or []
        rec["success"] = bool(res.get("success"))
        rec["turns"] = int(trace.get("turns") or len(steps) or 0)
        rec["tokens"] = int(trace.get("tokens") or trace.get("total_tokens") or 0)
        rec["tool_calls"] = len(steps)
        rec["steps_failed"] = sum(1 for s in steps if not s.get("success"))
        # 工具调用序列 —— 诊断"为什么这个任务花了 6 步"必须能看到它调了什么
        rec["tools"] = [
            (s.get("tool") or "?") + ("" if s.get("success", True) else "✗")
            for s in steps
        ]
        rec["answer_preview"] = str(res.get("result") or "")[:160]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed"] = round(time.time() - t0, 1)
    rec["degradations"] = _diff_degradations(before, _snapshot_degradations())
    return rec


def summarize(records: list[dict]) -> dict:
    n = len(records) or 1
    total_calls = sum(r["tool_calls"] for r in records)
    failed_calls = sum(r["steps_failed"] for r in records)
    deg = {}
    for r in records:
        for k, v in (r["degradations"] or {}).items():
            deg[k] = deg.get(k, 0) + v
    return {
        "tasks": len(records),
        "completion_rate": round(sum(1 for r in records if r["success"]) / n, 4),
        "avg_turns": round(sum(r["turns"] for r in records) / n, 2),
        "total_tool_calls": total_calls,
        "avg_tool_calls": round(total_calls / n, 2),
        # 无效调用 = 失败的工具调用 / 总调用。衡量"有没有空转"。
        "wasted_call_ratio": round(failed_calls / total_calls, 4) if total_calls else 0.0,
        "degradations": deg,
        "avg_elapsed_s": round(sum(r["elapsed"] for r in records) / n, 1),
        "total_tokens": sum(r["tokens"] for r in records),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务 (省成本)")
    ap.add_argument("--baseline", action="store_true", help="与上次结果对比")
    args = ap.parse_args()

    from config import DEEPSEEK_API_KEY

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-key-here":
        print("❌ 缺少 DEEPSEEK_API_KEY，请先配置 .env")
        return 1

    tasks = TASKS[: args.limit] if args.limit else TASKS

    # 建一个临时课题 + 极小知识库 (不依赖线上数据, 可重复运行)
    from config import LANCEDB_DIR
    from downloader import chunk_text_paged, parse_pdf
    from knowledge_base import build_knowledge_base
    from models import PaperMeta

    topic_id = f"agent_eval_{uuid.uuid4().hex[:8]}"
    table_name = f"papers_{topic_id}"
    pdfs = sorted(Path("paper_cache").glob("*.pdf"))
    if not pdfs:
        print("❌ paper_cache/ 下没有 PDF，无法建临时知识库")
        return 1

    data = pdfs[0].read_bytes()
    parsed = parse_pdf(data, pdfs[0].name)
    chunks = chunk_text_paged(
        parsed["full_text"],
        parsed.get("page_offsets") or [],
        sections=parsed.get("sections") or [],
    )
    import hashlib

    pid = "c" + hashlib.sha1(data).hexdigest()[:15]
    await build_knowledge_base(
        [PaperMeta(paper_id=pid, title="评测用论文", authors="", year=2024, is_oa=True)],
        {pid: chunks},
        topic_id,
    )
    print(f"临时知识库已建: {table_name} ({len(chunks)} 块, 源={pdfs[0].name})")

    records = []
    try:
        for i, t in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {t['id']} … ", end="", flush=True)
            rec = await run_one(t, topic_id, table_name)
            records.append(rec)
            print(
                f"{'✅' if rec['success'] else '❌'} "
                f"{rec['turns']} 步 / {rec['tool_calls']} 调用 / {rec['elapsed']}s"
                + (f" / {rec['error']}" if rec["error"] else "")
            )
            if rec["tools"]:
                print(f"      工具序列: {rec['tools']}")
    finally:
        try:
            import lancedb

            db = lancedb.connect(str(LANCEDB_DIR))
            db.drop_table(table_name)
            print(f"已清理临时表 {table_name}")
        except Exception as e:
            print(f"清理失败(需手工删): {e}")

    summary = summarize(records)
    report = {"summary": summary, "records": records}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.baseline and REPORT_PATH.exists():
        try:
            prev = json.loads(REPORT_PATH.read_text(encoding="utf-8")).get("summary", {})
            print("\n=== 与上次对比 ===")
            for k in ("completion_rate", "avg_turns", "avg_tool_calls", "wasted_call_ratio"):
                a, b = prev.get(k), summary.get(k)
                if a is None or b is None:
                    continue
                # 完成率越高越好; 其余越低越好
                better = (b > a) if k == "completion_rate" else (b < a)
                print(f"  {k:<20} {a} → {b}  {'✅' if better else '⚠️'}")
        except Exception as e:
            print(f"基线对比失败: {e}")

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n===== Agent 级评测汇总 =====")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")
    print(f"\n报告已保存: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
