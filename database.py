"""SQLite 数据库模块：替代 JSON 文件持久化，支持并发写入和事务。

表结构：
  - users: 用户注册/认证
  - topics: 课题持久化（替代 topics.json）
  - messages: 对话记录（替代 message_store 内存字典）
  - memory: ConversationMemory 磁盘存储（替代 data/memory/*.json）

用法：
    from database import get_db
    db = get_db()
    db.register_user("alice", hash, "a@b.com")
"""

import atexit
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/paperlink.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 线程锁: 保护表初始化和单例创建
_init_lock = threading.Lock()


class Database:
    """SQLite 数据库封装。每个线程独立连接，支持 WAL 模式并发。

    v5.2 修复 (DeepSeek 对话审查):
      - 建表只执行一次 (_tables_ready 标志), 避免高频操作反复执行 DDL
      - atexit 注册连接清理, 防连接泄漏
      - transaction() 上下文管理器, 支持原子事务
      - 反序列化失败记录日志
      - list_topics_full 支持 user_id 隔离
    """

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._local = threading.local()
        self._tables_ready = False
        atexit.register(self.close)

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # timeout=30: 并发写时等待锁, 避免 immediate 报错
            conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return self._local.conn

    def close(self):
        """关闭当前线程的连接。注册到 atexit, 进程退出时清理。"""
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def transaction(self):
        """事务上下文管理器: with db.transaction(): 内操作自动提交/回滚。

        用法:
            with db.transaction() as c:
                c.execute(...)
                c.execute(...)
        """
        return _Transaction(self)

    def _ensure_tables(self):
        # 只初始化一次 (DeepSeek 建议: 避免高频操作反复执行 DDL)
        if self._tables_ready:
            return
        with _init_lock:
            if self._tables_ready:
                return
            c = self._conn()
            c.executescript(
                """
            CREATE TABLE IF NOT EXISTS users (
                username    TEXT PRIMARY KEY,
                password    TEXT NOT NULL,
                email       TEXT DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS topics (
                topic_id        TEXT PRIMARY KEY,
                query           TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'building',
                lancedb_table   TEXT DEFAULT '',
                total_papers    INTEGER DEFAULT 0,
                uploaded_papers INTEGER DEFAULT 0,
                total_images    INTEGER DEFAULT 0,
                search_strategy TEXT DEFAULT NULL,
                step            TEXT DEFAULT '',
                current         INTEGER DEFAULT 0,
                dl_total        INTEGER DEFAULT 0,
                user_id         TEXT DEFAULT NULL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id    TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS memories (
                topic_id    TEXT PRIMARY KEY,
                recent      TEXT DEFAULT '[]',
                summary     TEXT DEFAULT '',
                core        TEXT DEFAULT '{}',
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- 人工标注队列: bad case 自动回流 → 人工校验 → 标注结果沉淀。
            -- 触发来源: 用户点踩 / 低置信度 / 无引用答案 (见 api 的 /feedback 与 enqueue)。
            CREATE TABLE IF NOT EXISTS review_queue (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT NOT NULL,
                answer      TEXT DEFAULT '',
                contexts    TEXT DEFAULT '[]',
                source      TEXT DEFAULT 'user_feedback',
                reason      TEXT DEFAULT '',
                user_id     TEXT DEFAULT '',
                status      TEXT NOT NULL DEFAULT 'pending',  -- pending | labeled | skipped
                label       TEXT DEFAULT '',                  -- 人工判定
                label_note  TEXT DEFAULT '',
                labeled_by  TEXT DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                labeled_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS learning_feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                rating      INTEGER NOT NULL DEFAULT 0,
                comment     TEXT DEFAULT '',
                user_id     TEXT DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS agent_skill_memory (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                task          TEXT NOT NULL DEFAULT '',
                tools_used    TEXT DEFAULT '',
                result_preview TEXT DEFAULT '',
                user_id       TEXT DEFAULT '',
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS team_runs (
                run_id       TEXT PRIMARY KEY,
                task         TEXT NOT NULL DEFAULT '',
                state_json   TEXT DEFAULT '{}',
                history_json TEXT DEFAULT '[]',
                current_stage TEXT DEFAULT '',
                pending_json TEXT DEFAULT NULL,
                status       TEXT DEFAULT 'active',
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Agent 中断快照: 暂停等待人工确认时落盘, 进程重启后仍可恢复。
            -- 此前该状态只存在 PiAgent 实例上 (_pi_agents 字典), 重启即丢 —— 用户点了
            -- 确认却发现"没有待确认的工具调用", 只能重跑整个任务 (已烧掉的 token 白费)。
            CREATE TABLE IF NOT EXISTS agent_pending_runs (
                run_id      TEXT PRIMARY KEY,
                user_id     TEXT DEFAULT '',
                goal        TEXT DEFAULT '',
                messages    TEXT DEFAULT '[]',
                state       TEXT DEFAULT '{}',
                pending     TEXT DEFAULT '[]',
                meta        TEXT DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS topic_state_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id     TEXT NOT NULL,
                from_status  TEXT NOT NULL DEFAULT '',
                to_status    TEXT NOT NULL,
                user_id      TEXT NOT NULL DEFAULT '',
                reason       TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_state_log_topic ON topic_state_log(topic_id);

            CREATE INDEX IF NOT EXISTS idx_messages_topic ON messages(topic_id);
            CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id);
        """
            )
            # 存量库迁移 (幂等): 老库无 user_id 列 → 补上, 否则多用户记忆串味
            for _tbl in ("learning_feedback", "agent_skill_memory"):
                try:
                    _cols = [
                        r[1] for r in c.execute(f"PRAGMA table_info({_tbl})").fetchall()
                    ]
                    if "user_id" not in _cols:
                        c.execute(
                            f"ALTER TABLE {_tbl} ADD COLUMN user_id TEXT DEFAULT ''"
                        )
                        logger.info(f"migrated: added user_id to {_tbl}")
                except Exception as e:
                    logger.warning(f"migrate {_tbl} failed: {e}")
            c.commit()
            self._tables_ready = True

    # ==================== Users ====================

    def register_user(
        self, username: str, password_hash: str, email: str = ""
    ) -> tuple[bool, str]:
        self._ensure_tables()
        c = self._conn()
        try:
            c.execute(
                "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                (username, password_hash, email),
            )
            c.commit()
            return True, "注册成功"
        except sqlite3.IntegrityError:
            return False, "用户名已存在"

    def get_user(self, username: str) -> Optional[dict]:
        self._ensure_tables()
        c = self._conn()
        row = c.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None

    # ==================== Topics ====================

    def save_topic(
        self,
        topic_id: str,
        query: str,
        status: str,
        lancedb_table: str = "",
        total_papers: int = 0,
        uploaded_papers: int = 0,
        total_images: int = 0,
        search_strategy: dict = None,
        step: str = "",
        current: int = 0,
        dl_total: int = 0,
        user_id: str = None,
    ):
        self._ensure_tables()
        c = self._conn()
        strat_json = (
            json.dumps(search_strategy, ensure_ascii=False) if search_strategy else None
        )
        c.execute(
            """
            INSERT INTO topics (topic_id, query, status, lancedb_table, total_papers,
                uploaded_papers, total_images, search_strategy, step, current, dl_total, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id) DO UPDATE SET
                query=excluded.query, status=excluded.status,
                lancedb_table=excluded.lancedb_table, total_papers=excluded.total_papers,
                uploaded_papers=excluded.uploaded_papers, total_images=excluded.total_images,
                search_strategy=excluded.search_strategy, step=excluded.step,
                current=excluded.current, dl_total=excluded.dl_total,
                updated_at=?
        """,
            (
                topic_id,
                query,
                status,
                lancedb_table,
                total_papers,
                uploaded_papers,
                total_images,
                strat_json,
                step,
                current,
                dl_total,
                user_id,
                datetime.now().isoformat(),
            ),
        )
        c.commit()

    def get_topic(self, topic_id: str) -> Optional[dict]:
        self._ensure_tables()
        c = self._conn()
        row = c.execute(
            "SELECT * FROM topics WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("search_strategy"):
            try:
                d["search_strategy"] = json.loads(d["search_strategy"])
            except Exception as e:
                logger.warning(f"Failed to parse search_strategy for {topic_id}: {e}")
        return d

    def list_topics(self, user_id: str = None) -> list[dict]:
        self._ensure_tables()
        c = self._conn()
        if user_id is not None:
            # 登录用户: 自己的课题 + 历史匿名(共享)课题; 匿名旧数据仍可见, 不丢历史
            rows = c.execute(
                "SELECT topic_id, query, status, total_papers, user_id, created_at FROM topics WHERE user_id = ? OR user_id IS NULL ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        else:
            # 匿名: 仅共享/匿名课题 (当前所有历史数据均为 NULL, 行为不变)
            rows = c.execute(
                "SELECT topic_id, query, status, total_papers, user_id, created_at FROM topics WHERE user_id IS NULL ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_topics_full(self, user_id: str = None) -> list[dict]:
        """全字段查询 — 供 _load_topics 恢复完整状态。
        v5.2: 支持 user_id 隔离 (DeepSeek 建议: 启动加载不泄漏其他用户课题)。"""
        self._ensure_tables()
        c = self._conn()
        if user_id is not None:
            rows = c.execute(
                "SELECT * FROM topics WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM topics ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_topic(self, topic_id: str) -> bool:
        self._ensure_tables()
        c = self._conn()
        cur = c.execute("DELETE FROM topics WHERE topic_id = ?", (topic_id,))
        c.commit()
        return cur.rowcount > 0

    def prune_archived_memories(self, days: int = 90) -> int:
        """记忆数据层过期: 清理已结题/中止/归档课题的过期记忆行。返回删除条数。

        此前只有**内存**层 TTL (释放进程缓存), 持久化记忆永不清理 → 长期膨胀。
        days<=0 表示不清理。
        """
        if days <= 0:
            return 0
        self._ensure_tables()
        c = self._conn()
        cur = c.execute(
            "DELETE FROM memories WHERE topic_id IN ("
            "  SELECT topic_id FROM topics "
            "  WHERE status IN ('settled', 'aborted', 'archived')"
            ") AND updated_at < datetime('now', ?)",
            (f"-{int(days)} days",),
        )
        c.commit()
        if cur.rowcount:
            logger.info(f"pruned {cur.rowcount} archived memories (>{days}d)")
        return cur.rowcount

    def log_state_change(
        self, topic_id: str, from_status: str, to_status: str,
        user_id: str = "", reason: str = "",
    ) -> None:
        """课题状态流转审计日志 (状态机: 谁/何时/从哪到哪/原因)。"""
        self._ensure_tables()
        c = self._conn()
        c.execute(
            "INSERT INTO topic_state_log (topic_id, from_status, to_status, user_id, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (topic_id, from_status, to_status, user_id, reason),
        )
        c.commit()

    def list_state_log(self, topic_id: str, limit: int = 50) -> list[dict]:
        """查询课题状态流转历史 (审计)。"""
        self._ensure_tables()
        c = self._conn()
        rows = c.execute(
            "SELECT * FROM topic_state_log WHERE topic_id = ? ORDER BY id DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ==================== Messages ====================

    def add_message(self, topic_id: str, role: str, content: str):
        self._ensure_tables()
        c = self._conn()
        c.execute(
            "INSERT INTO messages (topic_id, role, content) VALUES (?, ?, ?)",
            (topic_id, role, content),
        )
        c.commit()

    def get_messages(self, topic_id: str) -> list[dict]:
        self._ensure_tables()
        c = self._conn()
        rows = c.execute(
            "SELECT role, content, created_at FROM messages WHERE topic_id = ? ORDER BY id",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ==================== 人工标注队列 (bad case 回流) ====================

    def enqueue_review(
        self,
        question: str,
        answer: str = "",
        contexts: list | None = None,
        source: str = "user_feedback",
        reason: str = "",
        user_id: str = "",
    ) -> int:
        """把一个 bad case 放进人工标注队列。同一问题+来源不重复入队 (去重)。"""
        self._ensure_tables()
        c = self._conn()
        dup = c.execute(
            "SELECT id FROM review_queue WHERE question = ? AND source = ? "
            "AND status = 'pending' LIMIT 1",
            (question, source),
        ).fetchone()
        if dup:
            return dup["id"]
        cur = c.execute(
            """INSERT INTO review_queue
               (question, answer, contexts, source, reason, user_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                question,
                (answer or "")[:8000],
                json.dumps(contexts or [], ensure_ascii=False)[:20000],
                source,
                reason,
                user_id or "",
            ),
        )
        c.commit()
        return cur.lastrowid

    def list_review_queue(self, status: str = "pending", limit: int = 50) -> list[dict]:
        self._ensure_tables()
        c = self._conn()
        rows = c.execute(
            "SELECT * FROM review_queue WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["contexts"] = json.loads(d.get("contexts") or "[]")
            except (TypeError, ValueError):
                d["contexts"] = []
            out.append(d)
        return out

    def label_review(
        self, item_id: int, label: str, note: str = "", labeled_by: str = ""
    ) -> bool:
        """人工标注一条。

        label 分类 (与"规划错误/工具选错/参数错误单独归类"对应, 便于分别迭代):
          grounded        有据可依
          hallucinated    编造内容
          retrieval_miss  库里有但没召回
          bad_question    问题本身有问题
          tool_misuse     工具选错/该调没调
          param_error     工具参数写错
          planning_error  多步规划跑偏 (顺序错、漏步骤、绕远路)
        """
        self._ensure_tables()
        c = self._conn()
        cur = c.execute(
            """UPDATE review_queue
               SET status='labeled', label=?, label_note=?, labeled_by=?,
                   labeled_at=datetime('now')
               WHERE id=? AND status='pending'""",
            (label, note, labeled_by, item_id),
        )
        c.commit()
        return cur.rowcount > 0

    def review_stats(self) -> dict:
        self._ensure_tables()
        c = self._conn()
        by_status = {
            r["status"]: r["n"]
            for r in c.execute(
                "SELECT status, COUNT(*) n FROM review_queue GROUP BY status"
            ).fetchall()
        }
        by_label = {
            r["label"]: r["n"]
            for r in c.execute(
                "SELECT label, COUNT(*) n FROM review_queue "
                "WHERE status='labeled' AND label != '' GROUP BY label"
            ).fetchall()
        }
        by_source = {
            r["source"]: r["n"]
            for r in c.execute(
                "SELECT source, COUNT(*) n FROM review_queue GROUP BY source"
            ).fetchall()
        }
        return {
            "pending": by_status.get("pending", 0),
            "labeled": by_status.get("labeled", 0),
            "skipped": by_status.get("skipped", 0),
            "by_label": by_label,
            "by_source": by_source,
        }

    # ==================== Agent 中断快照 ====================

    def save_pending_run(
        self,
        run_id: str,
        user_id: str,
        goal: str,
        messages: list,
        state: dict,
        pending: list,
        meta: dict,
    ) -> None:
        """落盘中断快照 (暂停等待人工确认时调用)。同一 run_id 覆盖更新。"""
        self._ensure_tables()
        c = self._conn()
        c.execute(
            """INSERT INTO agent_pending_runs
               (run_id, user_id, goal, messages, state, pending, meta, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(run_id) DO UPDATE SET
                 user_id=excluded.user_id, goal=excluded.goal,
                 messages=excluded.messages, state=excluded.state,
                 pending=excluded.pending, meta=excluded.meta,
                 updated_at=datetime('now')""",
            (
                run_id,
                user_id or "",
                goal or "",
                json.dumps(messages, ensure_ascii=False),
                json.dumps(state, ensure_ascii=False),
                json.dumps(pending, ensure_ascii=False),
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        c.commit()

    def get_pending_run(self, run_id: str) -> Optional[dict]:
        """读回中断快照; 不存在返回 None。"""
        self._ensure_tables()
        c = self._conn()
        row = c.execute(
            "SELECT * FROM agent_pending_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for k, default in (
            ("messages", []),
            ("state", {}),
            ("pending", []),
            ("meta", {}),
        ):
            try:
                d[k] = json.loads(d.get(k) or "")
            except (TypeError, ValueError):
                d[k] = default
        return d

    def clear_pending_run(self, run_id: str) -> None:
        """恢复/放弃后清除快照, 避免陈旧数据被误恢复。"""
        self._ensure_tables()
        c = self._conn()
        c.execute("DELETE FROM agent_pending_runs WHERE run_id = ?", (run_id,))
        c.commit()

    # ==================== Memory ====================

    def save_memory(self, topic_id: str, recent: list, summary: str, core: dict):
        self._ensure_tables()
        c = self._conn()
        c.execute(
            "INSERT INTO memories (topic_id, recent, summary, core) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(topic_id) DO UPDATE SET"
            " recent=excluded.recent, summary=excluded.summary,"
            " core=excluded.core, updated_at=?",
            (
                topic_id,
                json.dumps(recent, ensure_ascii=False),
                summary,
                json.dumps(core, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )
        c.commit()

    def get_memory(self, topic_id: str) -> Optional[dict]:
        self._ensure_tables()
        c = self._conn()
        row = c.execute(
            "SELECT * FROM memories WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["recent"] = json.loads(d["recent"])
            d["core"] = json.loads(d["core"])
        except Exception as e:
            logger.warning(f"Failed to parse memory for {topic_id}: {e}")
            d["recent"], d["core"] = [], {}
        return d

    # ==================== Feedback (飞轮) ====================

    def record_feedback(
        self, question: str, answer: str, rating: int, comment: str = "",
        user_id: str = "",
    ):
        self._ensure_tables()
        c = self._conn()
        c.execute(
            "INSERT INTO learning_feedback (question, answer, rating, comment, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (question, answer, rating, comment, user_id or ""),
        )
        c.commit()

    def get_feedback_stats(self) -> dict:
        self._ensure_tables()
        c = self._conn()
        total = c.execute("SELECT COUNT(*) FROM learning_feedback").fetchone()[0]
        avg = c.execute("SELECT AVG(rating) FROM learning_feedback").fetchone()[0]
        return {"total": total, "avg_rating": round(avg, 2) if avg else 0.0}

    def get_low_rated_feedback(self, limit: int = 10, user_id: str = "") -> list[dict]:
        """低分反馈 (rating<=2) — 反馈驱动调整的燃料 (P2)。

        user_id 非空 → 只取该用户的反馈 (多用户隔离, 防 A 的教训注入 B 的提示)。"""
        self._ensure_tables()
        c = self._conn()
        if user_id:
            rows = c.execute(
                "SELECT question, comment, rating FROM learning_feedback "
                "WHERE rating <= 2 AND user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT question, comment, rating FROM learning_feedback "
                "WHERE rating <= 2 ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def record_skill_memory(
        self, task: str, tools_used: str, result_preview: str = "", user_id: str = ""
    ):
        """持久技能记忆 (P3): 记录成功任务的做法, 跨会话复用。保留最新 100 条。

        user_id: 多用户隔离 — 成功策略只在同一用户内复现。"""
        self._ensure_tables()
        c = self._conn()
        c.execute(
            "INSERT INTO agent_skill_memory (task, tools_used, result_preview, user_id) "
            "VALUES (?, ?, ?, ?)",
            (task[:200], tools_used[:200], result_preview[:300], user_id or ""),
        )
        # 只保留最新 100 条, 防无限增长
        c.execute(
            "DELETE FROM agent_skill_memory WHERE id NOT IN "
            "(SELECT id FROM agent_skill_memory ORDER BY id DESC LIMIT 100)"
        )
        c.commit()

    def get_skill_memory(self, limit: int = 100, user_id: str = "") -> list[dict]:
        """成功策略记忆。user_id 非空 → 只取该用户的策略 (多用户隔离)。"""
        self._ensure_tables()
        c = self._conn()
        if user_id:
            rows = c.execute(
                "SELECT task, tools_used, result_preview FROM agent_skill_memory "
                "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT task, tools_used, result_preview FROM agent_skill_memory "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


    # ==================== Team Runs (科研团队快照) ====================

    def save_team_run(
        self,
        run_id: str,
        task: str,
        state: dict,
        history: list,
        current_stage: str,
        pending_approval: dict = None,
        status: str = "active",
    ):
        """持久化科研团队快照 (DeepSeek 审查 #2: 崩溃可恢复, 多 run 并存)。"""
        self._ensure_tables()
        c = self._conn()
        c.execute(
            """
            INSERT INTO team_runs (run_id, task, state_json, history_json,
                current_stage, pending_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                task=excluded.task, state_json=excluded.state_json,
                history_json=excluded.history_json,
                current_stage=excluded.current_stage,
                pending_json=excluded.pending_json, status=excluded.status,
                updated_at=?
            """,
            (
                run_id,
                task,
                json.dumps(state, ensure_ascii=False),
                json.dumps(history, ensure_ascii=False),
                current_stage,
                json.dumps(pending_approval, ensure_ascii=False)
                if pending_approval
                else None,
                status,
                datetime.now().isoformat(),
            ),
        )
        c.commit()

    def get_team_run(self, run_id: str) -> Optional[dict]:
        """读取团队快照。返回 {task, state, history, current_stage, pending_approval, status}。"""
        self._ensure_tables()
        c = self._conn()
        row = c.execute(
            "SELECT * FROM team_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for col, target in (
            ("state_json", "state"),
            ("history_json", "history"),
        ):
            try:
                d[target] = json.loads(d[col])
            except Exception as e:
                logger.warning(f"Failed to parse {col} for run {run_id}: {e}")
                d[target] = {} if target == "state" else []
        try:
            d["pending_approval"] = (
                json.loads(d["pending_json"]) if d.get("pending_json") else None
            )
        except Exception:
            d["pending_approval"] = None
        return d


class _Transaction:
    """事务上下文管理器 (DeepSeek 建议: 多操作原子性)。

    用法:
        with db.transaction() as c:
            c.execute("INSERT INTO ...", ...)
            c.execute("UPDATE ...", ...)
    内任一操作失败 → 回滚全部; 成功 → 自动提交。
    """

    def __init__(self, db: Database):
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        c = self.db._conn()
        c.execute("BEGIN")
        return c

    def __exit__(self, exc_type, exc, tb):
        c = self.db._conn()
        if exc_type is None:
            c.commit()
        else:
            c.rollback()
        return False  # 不吞异常


# 全局单例
_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
