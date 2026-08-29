"""生成后接地增强 (post-hoc grounding pass): 拆论断 → 不支撑的查库补证据 → 重生成。

目标: 提高答案 grounding (faithfulness)。原理:
  1. 首轮生成答案
  2. 拆成论断, 对当前检索上下文判定支撑
  3. 不支撑的论断: 用论断本身查全库; 若找到支撑段落 → 加入证据池
  4. 证据池非空 → 用【原文+补充证据】重生成一次 (有界 1 次)
  5. 仍不支撑的论断 (库外参数知识) 在最终答案中显式标注 (库外通用知识)

代价: 拆论断(1 LLM) + 每条不支撑论断查库判定 + 重生成(1 LLM) ≈ +15-20s/答案。
默认关 (ask_question grounding=False), 评估路径可开启。
"""
import logging

logger = logging.getLogger(__name__)


def _split_claims(answer: str) -> list[str]:
    from claim_diagnose import _split_claims as _sc

    return _sc(answer)


def _judge_support(claim: str, contexts: list[str]) -> str:
    from claim_diagnose import _judge_support as _js

    return _js(claim, contexts)


def _retrieve_kb(table_name: str, claim: str, k: int = 2) -> list[dict]:
    from claim_diagnose import _retrieve_kb as _rk

    return _rk(table_name, claim, k)


async def grounding_pass(
    table_name: str, question: str, answer: str, base_contexts: list[dict]
) -> tuple[str, list[dict], list[str]]:
    """对已生成答案做接地增强。返回 (重生成答案, 扩展后上下文, 库外标注论断)。

    base_contexts: 首轮检索到的 chunk 行 (dict, 含 text/title 等)。"""
    claims = _split_claims(answer)
    if not claims:
        return answer, base_contexts, []

    # 1) 判定哪些论断不被当前上下文支撑
    ctx_texts = [c.get("text", "")[:900] for c in base_contexts[:8]]
    extra_rows: list[dict] = []
    unsupported_claims = []
    seen_ids = {c.get("chunk_id", "") for c in base_contexts}
    for cl in claims:
        rel = _judge_support(cl, ctx_texts)
        if rel == "supported":
            continue
        if rel == "contradicted":
            unsupported_claims.append(cl)  # 矛盾 — 标记库外/冲突
            continue
        # unaddressed → 查库
        rows = _retrieve_kb(table_name, cl, k=2)
        if not rows:
            unsupported_claims.append(cl)
            continue
        rows = [r for r in rows if r.get("chunk_id", "") not in seen_ids]
        if not rows:
            unsupported_claims.append(cl)
            continue
        # 用论断查到的段落判定是否真支撑
        if _judge_support(cl, [r.get("text", "")[:900] for r in rows]) == "supported":
            seen_ids.add(rows[0]["chunk_id"])
            extra_rows.append(rows[0])
        else:
            unsupported_claims.append(cl)

    if not extra_rows:
        return answer, base_contexts, unsupported_claims

    # 2) 用【原文+补充证据】重生成 (有界 1 次): 补充证据注入 prompt
    try:
        from qa_engine import ask_question

        new_ctx = base_contexts + extra_rows
        result = await ask_question(
            table_name, question, use_rerank=True, extra_context=extra_rows
        )
        new_answer = result.answer
        return new_answer, new_ctx, unsupported_claims
    except Exception as e:
        logger.warning(f"grounding regen failed: {e}")
        return answer, base_contexts + extra_rows, unsupported_claims
