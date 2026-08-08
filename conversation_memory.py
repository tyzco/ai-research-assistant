"""多轮对话记忆：SQLite 持久化 + 智能截断 + LLM摘要压缩。

准备2 状态管理方案：
  1. 固定状态模板留存核心变量（research_goal, key_findings, cited_papers）
  2. 智能截断保留高优先级内容（工具调用结果 > LLM思考 > 用户消息）
  3. 对话摘要压缩历史信息（超限时触发LLM摘要）

v5.2 修复 (DeepSeek 对话审查):
  - get_context 真 bug: 按优先级降序排序后取 [-N:] 导致保留最低优先级 + 丢时序
    → 改为取最近 N 条消息 (保持时间序)
  - _smart_truncate: summary 碎片拼接 → 待压缩缓冲区, 由 compress 一次性提炼
  - PRIORITY: 移除未使用的 assistant_summary
  - memories: TTL 惰性清理, 防内存泄漏
"""

import logging
import time

logger = logging.getLogger(__name__)

# 核心变量模板（Agent 状态追踪）
CORE_TEMPLATE = {
    "research_goal": "",
    "key_findings": [],  # [{content, source}]
    "cited_papers": [],
    "tools_used": [],
    "pending_questions": [],
}

# 内容优先级权重（越高越优先保留）— 仅用于截断决策, 不用于上下文排序
PRIORITY = {"tool_result": 10, "user": 5, "assistant": 3}

# 记忆缓存 TTL (秒) — 超过此时间未被访问则释放内存
MEMORY_TTL_SECONDS = 3600


class ConversationMemory:
    """管理单个课题的多轮对话记忆。SQLite 持久化。"""

    def __init__(self, topic_id: str, max_recent: int = None):
        if max_recent is None:
            from config import MEMORY_MAX_RECENT

            max_recent = MEMORY_MAX_RECENT
        self.topic_id = topic_id
        self.max_recent = max_recent
        self.recent: list[dict[str, str]] = []
        self.summary: str = ""
        self.core = dict(CORE_TEMPLATE)
        # 待压缩缓冲区: 从 recent 移出但尚未被 LLM 提炼的消息
        self._pending_compress: list[str] = []
        self._last_access = time.time()
        self._load()

    def _load(self):
        from database import get_db

        db = get_db()
        data = db.get_memory(self.topic_id)
        if data:
            self.recent = data.get("recent", [])
            self.summary = data.get("summary", "")
            self.core = data.get("core", dict(CORE_TEMPLATE))

    def save(self):
        from database import get_db

        db = get_db()
        db.save_memory(self.topic_id, self.recent, self.summary, self.core)

    def touch(self):
        """更新最后访问时间 (供 TTL 清理判断)。"""
        self._last_access = time.time()

    def update_core(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.core:
                if isinstance(self.core[k], list):
                    if v not in self.core[k]:
                        self.core[k].append(v)
                else:
                    self.core[k] = v
        self.save()

    def add_finding(self, content: str, source: str = ""):
        """结构化添加关键发现 (DeepSeek 建议: 保证 key_findings 元素是 dict)。
        去重按 content 字段比较, 而非整个 dict。"""
        if not content:
            return
        existing = {
            f.get("content", "")
            for f in self.core["key_findings"]
            if isinstance(f, dict)
        }
        if content not in existing:
            self.core["key_findings"].append({"content": content, "source": source})
            self.save()

    def add_turn(self, question: str, answer: str):
        self.recent.append({"role": "user", "content": question})
        self.recent.append({"role": "assistant", "content": answer})
        self._smart_truncate()
        self.save()

    def add_tool_call(self, tool_name: str, result: str):
        truncated = str(result)[:500]
        self.recent.append(
            {"role": "tool_result", "content": f"[{tool_name}] {truncated}"}
        )
        if tool_name not in self.core["tools_used"]:
            self.core["tools_used"].append(tool_name)
        self._smart_truncate()
        self.save()

    def _smart_truncate(self):
        """智能截断: 超过 max_recent*2 条时, 按优先级移除最低价值消息。
        被移除的消息进入待压缩缓冲区, 不直接碎片拼接进 summary。"""
        max_messages = self.max_recent * 2
        while len(self.recent) > max_messages:
            worst_idx, worst_score = 0, 99
            for i, m in enumerate(self.recent):
                s = PRIORITY.get(m.get("role", ""), 1)
                if s < worst_score:
                    worst_score, worst_idx = s, i
            removed = self.recent.pop(worst_idx)
            # 进入待压缩缓冲区 (保留完整内容供 LLM 提炼)
            self._pending_compress.append(removed.get("content", ""))

    async def compress(self):
        """LLM 摘要压缩: 将待压缩缓冲区的内容提炼进 summary。
        只在缓冲区达到阈值时才调用 (避免碎片拼接导致的冗余 LLM 请求)。"""
        if not self._pending_compress:
            return
        # 缓冲区过大或长期未压缩时才触发
        if len(self._pending_compress) < 3 and len(self.summary) < 200:
            return
        try:
            from openai import AsyncOpenAI

            from config import (
                DEEPSEEK_API_KEY,
                DEEPSEEK_BASE_URL,
                DEEPSEEK_MODEL,
                LLM_COREF_MAX_TOKENS,
                LLM_COREF_TEMPERATURE,
            )

            # 合并待压缩内容
            pending_text = "\n".join(self._pending_compress)[-2000:]
            to_compress = f"已有摘要: {self.summary}\n\n新内容: {pending_text}"

            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            resp = await client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=LLM_COREF_TEMPERATURE,
                max_tokens=LLM_COREF_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": f"将以下对话历史压缩为一段简短摘要（80字内）：\n{to_compress}",
                    }
                ],
            )
            self.summary = resp.choices[0].message.content.strip()[:200]
            self._pending_compress = []
            self.save()
        except Exception as e:
            logger.warning(f"Compression failed: {e}")

    def get_context(self) -> str:
        """构建上下文。保持时间序, 取最近 max_recent*2 条消息。
        v5.2 修复: 不再按优先级排序 (原实现丢失时序 + 保留最低优先级)。"""
        self.touch()
        parts = []
        if self.core["research_goal"]:
            parts.append(f"【调研目标】{self.core['research_goal']}")
        if self.core["key_findings"]:
            findings = [
                f.get("content", "")[:100]
                for f in self.core["key_findings"][-5:]
                if isinstance(f, dict)
            ]
            if findings:
                parts.append("【关键发现】" + "; ".join(findings))
        if self.summary:
            parts.append(f"【历史摘要】{self.summary}")

        # 保持原始时间序, 取最近的消息
        recent_slice = self.recent[-self.max_recent * 2 :]
        for m in recent_slice:
            role = {"user": "用户", "assistant": "AI", "tool_result": "工具"}.get(
                m["role"], m["role"]
            )
            parts.append(f"[{role}]: {m['content'][:300]}")
        return "\n".join(parts)

# ===== 记忆缓存 (TTL 惰性清理, 防内存泄漏) =====

memories: dict[str, ConversationMemory] = {}


def get_memory(topic_id: str) -> ConversationMemory:
    """获取记忆实例。带 TTL 惰性清理: 访问时清理过期条目。"""
    now = time.time()
    # 惰性清理: 仅在缓存较大时扫描 (避免每次 O(n))
    if len(memories) > 20:
        expired = [
            tid
            for tid, mem in memories.items()
            if now - mem._last_access > MEMORY_TTL_SECONDS
        ]
        for tid in expired:
            # 已持久化到 SQLite, 可安全释放内存
            memories.pop(tid, None)

    if topic_id not in memories:
        memories[topic_id] = ConversationMemory(topic_id)
    memories[topic_id].touch()
    return memories[topic_id]


def close_memory(topic_id: str):
    """显式关闭课题记忆, 释放内存 (持久化已完成, 从缓存移除)。"""
    memories.pop(topic_id, None)
