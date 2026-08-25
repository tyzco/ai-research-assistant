"""后台订阅调度器 — 定时自动执行论文检索。

服务启动时自动加载所有订阅，按 interval_hours 周期执行。

设计:
  - 单线程异步循环，不依赖 celery/redis
  - 每个订阅独立计时，互不阻塞
  - 执行记录写入 data/scheduler_log.json
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_FILE = Path("data/scheduler_log.json")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


async def _run_all_subscriptions_once():
    """执行所有订阅一次。"""
    from integrations import _load_subscriptions, _save_subscriptions
    from search import search_papers_for_topic

    data = _load_subscriptions()
    subs = data.get("subscriptions", [])
    if not subs:
        return

    log_entries = []
    for sub in subs:
        now = datetime.now()
        last = datetime.fromisoformat(sub["last_run"]) if sub.get("last_run") else None
        interval = sub.get("interval_hours", 24)

        # 跳过还没到时间的
        if last and (now - last).total_seconds() < interval * 3600:
            continue

        try:
            papers = await search_papers_for_topic(
                sub["query"],
                keywords_en=sub.get("keywords_en") or None,
                keywords_cn=sub.get("keywords_cn") or None,
            )
            sub["last_run"] = now.isoformat()
            sub["total_papers_found"] = sub.get("total_papers_found", 0) + len(papers)
            log_entries.append(
                {
                    "subscription_id": sub["id"],
                    "query": sub["query"],
                    "user_id": sub.get("user_id", ""),
                    "time": now.isoformat(),
                    "new_papers": len(papers),
                    "status": "ok",
                }
            )
            logger.info(
                f"Scheduler: {sub['id']} ({sub['query']}) found {len(papers)} papers"
            )
        except Exception as e:
            log_entries.append(
                {
                    "subscription_id": sub.get("id", "?"),
                    "query": sub.get("query", "?"),
                    "user_id": sub.get("user_id", ""),
                    "time": now.isoformat(),
                    "status": f"error: {e}",
                }
            )
            logger.warning(f"Scheduler: {sub.get('id')} failed: {e}")

    _save_subscriptions(data)

    # 追加日志
    existing = []
    if LOG_FILE.exists():
        try:
            existing = json.loads(LOG_FILE.read_text())
        except Exception:
            pass
    existing = (existing + log_entries)[-200:]  # 保留最近 200 条
    LOG_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


async def scheduler_loop(interval_seconds: int = 3600):
    """后台调度主循环，每小时检查一次。"""
    logger.info(f"Scheduler started (interval={interval_seconds}s)")
    while True:
        try:
            await _run_all_subscriptions_once()
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
        await asyncio.sleep(interval_seconds)


def start_scheduler(interval_seconds: int = 3600):
    """启动后台调度器 (asyncio.create_task, 非阻塞)。"""
    asyncio.create_task(scheduler_loop(interval_seconds))
    logger.info("Scheduler task created")
