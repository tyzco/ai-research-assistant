"""自动幻觉检测工具 — 答案断言 vs 证据一致性 (factcheck)。

放在 agent/tools_extra/ 下, 注册表启动时自动发现注册。
能力: check_factual_support(claim, evidence|table_name) → LLM 判定 supported/contradicted/unsupported。

边界 (诚实声明):
  LLM 判定不是 ground truth (它自己也可能有偏差), 这是「一致性复核」而非「事实裁决」。
  用途是给 agent 的生成答案做一层来源锚定检查, 降低幻觉风险, 不承诺 100% 准确。
"""
import json
import logging

logger = logging.getLogger(__name__)


async def _retrieve_evidence(claim: str, table_name: str) -> str:
    """从知识库检索证据文本 (top-3 chunks)。"""
    from hybrid_retriever import AdaptiveHybridRetriever

    hr = AdaptiveHybridRetriever(table_name)
    # 修复: 漏 await → 解包 coroutine 抛 TypeError → 事实核查工具**永远拿不到证据**(总是失败)
    results, _ = await hr.search(claim, top_k=3)
    parts = []
    for r in results:
        title = r.get("title", "N/A")
        text = (r.get("text") or "")[:300]
        if text.strip():
            parts.append(f"[{title}] {text}")
    return "\n\n".join(parts)


async def _check_factual_support(
    claim: str, evidence: str = "", table_name: str = ""
) -> dict:
    """自动幻觉检测: 判断「结论 claim」是否被「证据 evidence」支撑。

    evidence 与 table_name 二选一: 提供 evidence 直接用, 否则用 table_name 检索 KB 取证据。
    返回: {verdict: supported|contradicted|unsupported, reason, evidence_used}
    """
    claim = (claim or "").strip()
    if len(claim) < 10:
        return {"error": "结论太短，无法检测（至少 10 字符）"}

    # 取证据
    if not evidence and table_name:
        try:
            evidence = await _retrieve_evidence(claim, table_name)
        except Exception as e:
            return {"error": f"证据检索失败: {e}"}
    if not evidence or not evidence.strip():
        return {"error": "需要 evidence 参数 或 table_name (且知识库有内容)"}

    from config import DEEPSEEK_MODEL
    from llm_client import chat_with_retry

    try:
        resp = await chat_with_retry(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是学术事实核查员。判断「结论」是否被「证据」支撑。"
                        "只返回一行 JSON, 不要其他文字: "
                        '{"verdict":"supported|contradicted|unsupported","reason":"一句话理由"}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"结论:\n{claim}\n\n证据:\n{evidence[:3000]}",
                },
            ],
            temperature=0.0,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"error": f"LLM 判定失败: {e}"}

    # 解析 JSON (容错: 提取首个 {...})
    try:
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
        verdict = parsed.get("verdict", "unsupported")
        reason = parsed.get("reason", raw)
    except Exception:
        # 非 JSON → 从原文判断 (LLM 偶发不遵格式)
        low = raw.lower()
        verdict = (
            "supported"
            if "supported" in low
            else "contradicted"
            if "contradicted" in low
            else "unsupported"
        )
        reason = raw[:200]

    verdict = (
        verdict
        if verdict in ("supported", "contradicted", "unsupported")
        else "unsupported"
    )
    return {
        "verdict": verdict,
        "reason": reason[:200],
        "evidence_used": evidence[:500],
        "note": "LLM 一致性复核, 非事实裁决; 供 agent 自查, 不替代人工判断。",
    }


from agent.tool_registry import Tool  # noqa: E402  (必须放文件尾, 避免循环导入)

TOOLS = [
    Tool(
        "check_factual_support",
        "自动幻觉检测: 判断一句结论是否被证据支撑 (supported/contradicted/unsupported)。"
        "用于 agent 生成答案后自查, 降低编造引用/无据断言。evidence 与 table_name 二选一。",
        {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "要核查的结论/断言 (如生成答案里的某个论点)"},
                "evidence": {
                    "type": "string",
                    "description": "证据文本 (可选, 与 table_name 二选一)",
                },
                "table_name": {
                    "type": "string",
                    "description": "知识库表名 (可选, 从 KB 检索证据)",
                },
            },
            "required": ["claim"],
        },
        _check_factual_support,
        kind="llm"
    ),
]
