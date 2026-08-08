"""按量付费配额系统 — 免费额度 + 用量追踪 + 充值占位。

变现闭环 (能演示 → 能收钱):
  免费额度体验 → LLM 用量按用户追踪 → 超额提示充值 → top_up 充值 (占位)

真实支付 (Stripe/支付宝) 是外部集成, 此处留 top_up 入口, 支付回调接入后即可闭环。
匿名按 IP 记账, 登录按 user_id 记账。

自测: python quota.py
"""

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

_usage_file = Path("data/quota_usage.json")
_usage: dict = {}  # user_key -> {"tokens": int(已用), "paid": int(已充值额度), "updated": float}


def _load() -> None:
    global _usage
    try:
        if _usage_file.exists():
            _usage = json.loads(_usage_file.read_text())
    except Exception:
        _usage = {}


def _save() -> None:
    try:
        _usage_file.parent.mkdir(parents=True, exist_ok=True)
        _usage_file.write_text(json.dumps(_usage, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"quota save failed: {e}")


def get_user_key(user_id: str = "", ip: str = "") -> str:
    """匿名按 IP, 登录按 user_id。"""
    if user_id:
        return f"user:{user_id}"
    return f"anon:{ip or 'unknown'}"


def check_quota(user_key: str) -> dict:
    """查询配额状态。blocked=True → 需充值。"""
    from config import QUOTA_FREE_TOKENS, QUOTA_ENABLED

    if not QUOTA_ENABLED:
        return {"enabled": False, "allowed": None, "used": 0, "remaining": None, "blocked": False}
    entry = _usage.get(user_key, {})
    used = int(entry.get("tokens", 0))
    allowance = int(QUOTA_FREE_TOKENS) + int(entry.get("paid", 0))
    remaining = max(0, allowance - used)
    return {
        "enabled": True,
        "allowed": allowance,
        "used": used,
        "remaining": remaining,
        "blocked": remaining <= 0,
    }


def record_usage(user_key: str, tokens: int) -> None:
    """记录用量 (LLM prompt+completion token 数)。"""
    if not tokens or tokens <= 0:
        return
    _load()
    entry = _usage.setdefault(user_key, {"tokens": 0, "paid": 0, "updated": time.time()})
    entry["tokens"] = int(entry.get("tokens", 0)) + int(tokens)
    entry["updated"] = time.time()
    _save()


def top_up(user_key: str, tokens: int) -> dict:
    """充值占位: 增加可用额度 (真实支付回调后调用)。"""
    if tokens <= 0:
        return check_quota(user_key)
    _load()
    entry = _usage.setdefault(user_key, {"tokens": 0, "paid": 0, "updated": time.time()})
    entry["paid"] = int(entry.get("paid", 0)) + int(tokens)
    entry["updated"] = time.time()
    _save()
    logger.info(f"Top-up {user_key}: +{tokens} tokens")
    return check_quota(user_key)


_load()


def _selftest() -> None:
    import tempfile

    global _usage_file
    _usage_file = Path(tempfile.mktemp(suffix=".json"))
    os.environ["QUOTA_ENABLED"] = "true"
    os.environ["QUOTA_FREE_TOKENS"] = "1000"
    import config

    config.QUOTA_ENABLED = True
    config.QUOTA_FREE_TOKENS = 1000

    key = get_user_key(user_id="test_user")
    print("初始:", check_quota(key))
    record_usage(key, 600)
    print("用 600 后:", check_quota(key))
    record_usage(key, 500)
    print("再用 500 后 (超额):", check_quota(key))
    print("充值 2000:", top_up(key, 2000))
    print("充值后 remaining:", check_quota(key)["remaining"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
