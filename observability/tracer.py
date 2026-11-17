"""Langfuse 追踪 (可选, SDK v2 HTTP ingestion)。

边界:
  - 只做观测: 无 key / 连不上时全部 no-op, 绝不影响运行。
  - prompt management 只做**版本登记**, 运行时仍用代码里的提示词。
  - 自研 SQLite 仍是实时画布/回放的唯一数据源。
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_PROMPT_CACHE: set[str] = set()


def _client():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        return None
    try:
        from langfuse import Langfuse

        # 仅观测请求走代理; 不影响 DeepSeek/检索等其它 HTTP
        proxy = os.getenv("LANGFUSE_PROXY", "")
        httpx_client = None
        if proxy:
            import httpx

            httpx_client = httpx.Client(proxy=proxy, timeout=25)

        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
            environment=os.getenv("LANGFUSE_ENV", "local"),
            httpx_client=httpx_client,
            timeout=25,
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"langfuse init failed: {e}")
        return None


class RunTracer:
    """一次运行: trace + 每个 Agent 的 span + 工具/检索子 span。"""

    def __init__(self, run_id: str, task: str, params: dict, plan: list | None = None):
        self.lf = _client()
        self.run_id = run_id
        self.trace = None
        self.spans: dict[str, object] = {}
        self.tools: dict[str, object] = {}
        self.t0 = time.time()
        self._finished = False
        if not self.lf:
            return
        try:
            self.trace = self.lf.start_observation(
                name="team_run",
                as_type="chain",
                input={"task": task, "params": params},
                metadata={"run_id": run_id, "plan": plan or []},
            )
            self.trace_id = self.trace.trace_id
        except Exception as e:
            logger.warning(f"langfuse trace start failed: {e}")
            self.lf = None

    def register_prompts(self, prompts: dict[str, str]) -> None:
        """把各 Agent 的系统提示词登记为 prompt 版本 (仅首次)。

        放到后台线程: 每个 create_prompt 都是一次真实 HTTP 请求, 14 个串行
        走代理会阻塞整个运行 (实测把 SSE 拖到断流)。
        """
        if not self.lf:
            return
        import threading

        def _run():
            for key, text in prompts.items():
                name = f"paperlink/agent/{key}"
                if name in _PROMPT_CACHE:
                    continue
                try:
                    self.lf.create_prompt(
                        name=name,
                        prompt=text,
                        labels=["production"],
                        type="text",
                        commit_message="registered from research_team.py",
                    )
                    _PROMPT_CACHE.add(name)
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()

    def on_event(self, ev: dict) -> None:
        if not self.lf or self.trace is None:
            return
        try:
            self._dispatch(ev)
        except Exception:
            pass

    def _span(self, name: str, as_type: str = "span", **kwargs):
        try:
            from langfuse.types import TraceContext

            return self.lf.start_observation(
                name=name,
                as_type=as_type,
                trace_context=TraceContext(trace_id=self.trace_id),
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"langfuse span {name} failed: {e}")
            return None

    def _dispatch(self, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "stage":
            key = ev.get("agent") or ""
            if ev.get("status") == "running":
                self.spans[key] = self._span(
                    ev.get("name") or key, "span", input={"agent": key}
                )
            else:
                span = self.spans.pop(key, None)
                if span is not None:
                    span.update(output=str(ev.get("preview") or "")[:3000])
                    span.end()
        elif kind == "agent_tool":
            key = f"{ev.get('agent')}:{ev.get('tool')}"
            if ev.get("status") == "start":
                self.tools[key] = self._span(
                    str(ev.get("tool")), "tool",
                    input={"agent": ev.get("agent"), "args": ev.get("args")},
                )
            else:
                span = self.tools.pop(key, None)
                if span is not None:
                    span.update(
                        output=str(ev.get("result") or "")[:3000],
                        metadata={"ok": bool(ev.get("ok"))},
                    )
                    span.end()
        elif kind == "rag":
            span = self._span(f"rag:{ev.get('stage')}", "retriever", input=ev.get("detail"))
            if span is not None:
                span.update(output=ev.get("detail"))
                span.end()
        elif kind == "gate":
            self._span(
                "approval_gate", "guardrail",
                input={"agent": ev.get("agent"), "gate": ev.get("gate")},
            )
        elif kind in ("result", "team_done"):
            self.finish(str(ev.get("result") or ""), "completed")

    def finish(self, output: str, status: str) -> None:
        if not self.lf or self.trace is None or self._finished:
            return
        self._finished = True
        try:
            for span in list(self.spans.values()):
                span.end()
            for span in list(self.tools.values()):
                span.end()
            self.spans.clear()
            self.tools.clear()
            self.trace.update(
                output=str(output)[:4000],
                metadata={"status": status, "run_id": self.run_id},
            )
            self.trace.end()
            elapsed = (time.time() - self.t0) * 1000
            self.lf.create_score(trace_id=self.trace_id, name="duration_ms", value=round(elapsed, 1))
            self.lf.create_score(trace_id=self.trace_id, name="result_chars", value=len(output or ""))
            self.lf.flush()
        except Exception as e:
            logger.warning(f"langfuse finish failed: {e}")
