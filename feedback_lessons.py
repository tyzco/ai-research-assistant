"""反馈驱动调整 (P2) — 数据飞轮第二环。

从 learning_feedback 低分数据 (rating<=2) 提炼「易失败教训」, 注入 Agent 系统提示,
让后续回答规避用户已验证的失败类型。这是「自我进化」的安全路径: 数据驱动, 人类把关。

缓存: LLM 蒸馏结果带 TTL (反馈变化慢, 不必每次重算)。
安全: 只提炼「低分问题类型/改进方向」的抽象教训, 不把用户原始问答明文扩散到提示外。
"""

import json
import logging
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

_CACHES: dict[str, dict] = {}  # user_id → {"ts", "lessons"} — 按用户分键 (单键缓存会串味)
_TTL = 300  # 5 分钟


def _cache_for(user_id: str) -> dict:
    return _CACHES.setdefault(user_id or "", {"ts": 0.0, "lessons": []})


def _distill(rows: list[dict]) -> list[str]:
    """LLM 把低分反馈提炼成 2-3 条「易失败教训」。"""
    samples = "\n".join(
        f"- Q: {(r.get('question') or '')[:120]} | 评分: {r.get('rating')} | "
        f"评语: {(r.get('comment') or '')[:120]}"
        for r in rows
    )
    prompt = (
        "以下是从用户反馈中提取的低分问答样本（用户对回答不满意）。"
        "请归纳 2-3 条可执行的「易失败教训」，帮助 AI 助手以后避免同类错误。\n"
        "要求：具体、可执行、中文，每条一句话。只输出 JSON 数组，如 [\"教训1\", \"教训2\"]，不要解释。\n\n"
        f"低分样本:\n{samples}"
    )
    try:
        from openai import OpenAI

        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.2,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
            timeout=30,
        )
        import re as _re

        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        raw = _re.sub(r",\s*([}\]])", r"\1", raw)  # 清理尾随逗号
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            logger.warning(f"Lesson distill bad response: {raw[:100]}")
            return []
        arr = json.loads(raw[start : end + 1])
        return [str(x)[:120] for x in arr if isinstance(x, str)][:3]
    except Exception as e:
        logger.warning(f"Feedback lesson distillation failed: {e}")
        return []


def get_failure_lessons(force: bool = False, user_id: str = "") -> list[str]:
    """返回「易失败教训」列表 (空 = 无低分反馈或蒸馏不可用)。按用户 TTL 缓存。

    user_id 非空 → 只蒸馏该用户的低分反馈 (多用户隔离: A 的教训不注入 B 的提示)。
    """
    now = time.time()
    cache = _cache_for(user_id)
    if not force and now - cache["ts"] < _TTL:
        return cache["lessons"]
    try:
        from database import get_db

        rows = get_db().get_low_rated_feedback(limit=10, user_id=user_id)
        if len(rows) < 3:  # 样本太少不蒸馏, 避免过拟合噪声
            cache.update({"ts": now, "lessons": []})
            return []
        lessons = _distill(rows)
        cache.update({"ts": now, "lessons": lessons})
        if lessons:
            logger.info(f"Feedback lessons refreshed (user={user_id or 'anon'}): {lessons}")
        return lessons
    except Exception as e:
        logger.warning(f"get_failure_lessons failed: {e}")
        return []
