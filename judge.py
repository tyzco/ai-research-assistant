"""统一判断接口 (判断类 LLM 调用抽象层) — 对齐 TypeSafe Jev 的三原语。

背景: 项目的"判断类"调用散落在 scope_gate / draft_quality / skills 路由 / 接地验证等处,
各自拼 prompt + 解析 JSON。本模块收敛为统一接口, 使**判断模型可替换**:

    choice(state, question, options) → "从列表选一个"
    score(state, question, rubric)   → "按 rubric 打分"
    noul(state, statement)           → "这个陈述为真吗" (0-1)

现实现 DeepSeekJudge (prompt → 解析)。将来接入 Jev (类型化问题 API, 70-500ms/输出免费)
只需新增 JevJudge 实现并改 config.JUDGE_BACKEND, 判断点代码零改动。

为什么值得解耦: 判断类调用在关键路径上 (scope_gate 每请求都过、接地验证每答案都跑),
与生成类需求不同 — 判断要快/便宜/可校准, 生成要写得好。分开后换任一不影响另一条链路。
"""
import json
import logging

logger = logging.getLogger(__name__)


class JudgeResult:
    """判断结果: 值 + 置信度 (对齐 Jev 的 confidence 语义; DeepSeek 实现下置信度可能为空)。"""

    __slots__ = ("value", "confidence", "raw")

    def __init__(self, value=None, confidence: float = 0.0, raw: str = ""):
        self.value = value
        self.confidence = confidence
        self.raw = raw

    def __repr__(self):
        return f"JudgeResult(value={self.value!r}, confidence={self.confidence:.2f})"


class DeepSeekJudge:
    """DeepSeek 实现: prompt → 温度0 → 解析。保持项目既有约定 (thinking disabled)。"""

    def _call(self, prompt: str, max_tokens: int = 200, timeout: int = 30) -> str:
        from openai import OpenAI

        from config import (
            DEEPSEEK_API_KEY,
            DEEPSEEK_BASE_URL,
            DEEPSEEK_MODEL,
            LLM_QUICK_MAX_TOKENS,
        )

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            max_tokens=min(LLM_QUICK_MAX_TOKENS, max_tokens),
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""

    def choice(self, state: str, question: str, options: list[str]) -> JudgeResult:
        """从 options 里选一个 (对齐 Jev Choice 原语)。"""
        if not options:
            return JudgeResult(None, 0.0)
        opts = "\n".join(f"- {o}" for o in options)
        prompt = (
            f"你是判断器。根据给定状态回答问题, 从选项里选**恰好一个**。\n"
            f"状态:\n{(state or '')[:2000]}\n\n问题: {question}\n\n选项:\n{opts}\n\n"
            f"只输出选中的选项原文 (不要解释、不要多余文字)。"
        )
        raw = self._call(prompt, max_tokens=40)
        v = (raw or "").strip()
        for o in options:
            if o.strip() == v or o.strip() in v:
                return JudgeResult(o, 1.0, raw)
        return JudgeResult(options[0], 0.3, raw)  # 解析失败保守兜底

    def score(self, state: str, question: str, rubric: str = "", lo: int = 0, hi: int = 5) -> JudgeResult:
        """按 rubric 打分 (对齐 Jev Score 原语)。"""
        prompt = (
            f"你是评分器。根据 rubric 给状态打分, 输出 {lo}-{hi} 的整数。\n"
            f"rubric: {rubric}\n状态:\n{(state or '')[:2000]}\n\n问题: {question}\n\n只输出一个整数。"
        )
        raw = self._call(prompt, max_tokens=10)
        import re

        m = re.search(r"-?\d+", raw or "")
        if not m:
            return JudgeResult(None, 0.0, raw)
        return JudgeResult(max(lo, min(hi, int(m.group()))), 1.0, raw)

    def noul(self, state: str, statement: str) -> JudgeResult:
        """判断陈述是否为真 (对齐 Jev Noul 原语, 返回 0-1)。"""
        prompt = (
            f"你是判断器。仅依据给定状态, 判断下列陈述是否为真。\n"
            f"状态:\n{(state or '')[:2500]}\n\n陈述: {statement}\n\n"
            f"只输出一个 0-1 之间的小数 (1=明确为真, 0=明确为假; 状态未提及时给低分)。"
        )
        raw = self._call(prompt, max_tokens=10)
        import re

        m = re.search(r"\d+\.?\d*", raw or "")
        if not m:
            return JudgeResult(None, 0.0, raw)
        v = max(0.0, min(1.0, float(m.group())))
        return JudgeResult(v >= 0.5, v, raw)

    def noul_batch(self, state: str, statements: list[str]) -> list[JudgeResult]:
        """批量判断多条陈述 (对齐 Jev"多问题并行、加问题几乎不增延迟"的模式)。

        DeepSeek 无并行采样, 用一次调用 + JSON 数组输出近似 (比逐条调用省 N-1 次往返)。
        """
        if not statements:
            return []
        items = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        prompt = (
            f"你是判断器。仅依据给定状态, 逐条判断下列陈述是否为真 (0-1)。\n"
            f"状态:\n{(state or '')[:2500]}\n\n陈述列表:\n{items}\n\n"
            f'只输出 JSON 数组, 与陈述顺序一一对应, 如 [0.9, 0.1, 0.7]。不要解释。'
        )
        raw = self._call(prompt, max_tokens=min(600, 40 + 12 * len(statements)))
        try:
            arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
            out = []
            for i in range(len(statements)):
                v = float(arr[i]) if i < len(arr) else 0.0
                v = max(0.0, min(1.0, v))
                out.append(JudgeResult(v >= 0.5, v, raw))
            return out
        except Exception:
            return [JudgeResult(None, 0.0, raw) for _ in statements]


_judge = None


def get_judge():
    """判断器工厂。当前 DeepSeek 实现; 将来 JUDGE_BACKEND=jev 时返回 JevJudge。"""
    global _judge
    if _judge is None:
        _judge = DeepSeekJudge()
    return _judge


# ---- 项目判断点的便捷封装 (统一收敛处; 将来换 judge 后端即全部生效) ----

def judge_grounded(claim: str, context: str) -> JudgeResult:
    """接地验证: 论断是否被上下文支撑 (faithfulness 的核心判断)。"""
    return get_judge().noul(context, claim)


def judge_academic(query: str) -> JudgeResult:
    """学术白名单判断: 该请求是否属学术用途。"""
    return get_judge().noul(
        f"用户请求如下 (可能中英文):\n{query[:500]}",
        "这是一个与学术文献检索/论文写作/科研相关的请求",
    )


def judge_draft_items(draft: str, items: list[dict]) -> list[JudgeResult]:
    """初稿多项自检 (每项一个陈述, 批量判断)。"""
    statements = [f"{it.get('label','')}: {it.get('check','')}" for it in items]
    return get_judge().noul_batch(draft[:4000], statements)
