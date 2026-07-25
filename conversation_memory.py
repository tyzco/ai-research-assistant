"""多轮对话记忆：状态模板 + 智能截断 + LLM摘要压缩。

准备2 §状态管理方案：
  1. 固定状态模板留存核心变量（research_goal, key_findings, cited_papers）
  2. 智能截断保留高优先级内容（工具调用结果 > LLM思考 > 用户消息）
  3. 对话摘要压缩历史信息（max_recent*2超限时触发LLM摘要）"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("data/memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# 核心变量模板（Agent 状态追踪）
CORE_TEMPLATE = {
    "research_goal": "",  # 调研目标
    "key_findings": [],  # 关键发现 [{content, source}]
    "cited_papers": [],  # 已引用论文 [paper_id]
    "tools_used": [],  # 已调用工具 [tool_name]
    "pending_questions": [],  # 待回答问题
}

# 内容优先级权重（越高越优先保留）
PRIORITY = {"tool_result": 10, "assistant_summary": 9, "user": 5, "assistant": 3}


class ConversationMemory:
    """管理单个课题的多轮对话记忆。"""

    def __init__(self, topic_id: str, max_recent: int = 5):
        self.topic_id = topic_id
        self.max_recent = max_recent
        self.recent: list[dict[str, str]] = []
        self.summary: str = ""
        self.core = dict(CORE_TEMPLATE)  # 状态模板
        self._load()

    def _path(self):
        return MEMORY_DIR / f"{self.topic_id}.json"

    def _load(self):
        p = self._path()
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self.recent = data.get("recent", [])
                self.summary = data.get("summary", "")
                self.core = data.get("core", dict(CORE_TEMPLATE))
            except Exception:
                pass

    def save(self):
        self._path().write_text(
            json.dumps(
                {"recent": self.recent, "summary": self.summary, "core": self.core},
                ensure_ascii=False,
            )
        )

    def update_core(self, **kwargs):
        """更新核心状态变量。"""
        for k, v in kwargs.items():
            if k in self.core:
                if isinstance(self.core[k], list):
                    self.core[k].append(v) if v not in self.core[k] else None
                else:
                    self.core[k] = v
        self.save()

    def add_turn(self, question: str, answer: str):
        self.recent.append({"role": "user", "content": question})
        self.recent.append({"role": "assistant", "content": answer})
        self._smart_truncate()
        self.save()

    def add_tool_call(self, tool_name: str, result: str):
        """记录工具调用结果（高优先级保留）。"""
        truncated = str(result)[:500]
        self.recent.append(
            {"role": "tool_result", "content": f"[{tool_name}] {truncated}"}
        )
        self.core["tools_used"].append(tool_name)
        self._smart_truncate()
        self.save()

    def _smart_truncate(self):
        """智能截断：超限时按优先级淘汰低价值消息。"""
        max_messages = self.max_recent * 2
        while len(self.recent) > max_messages:
            # 找最低优先级的消息踢掉
            worst_idx, worst_score = 0, 99
            for i, m in enumerate(self.recent):
                s = PRIORITY.get(m.get("role", ""), 1)
                if s < worst_score:
                    worst_score, worst_idx = s, i
            removed = self.recent.pop(worst_idx)
            # 合并到摘要
            self.summary = (
                self.summary + " | " + removed.get("content", "")[:80]
            ).strip(" |")[-500:]

    async def compress(self):
        """LLM 摘要压缩：将当前摘要历史压缩为一段摘要。"""
        if len(self.summary) < 200:
            return
        try:
            from openai import AsyncOpenAI

            from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            resp = await client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=0.1,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": f"将以下对话历史压缩为一段简短摘要（80字内）：\n{self.summary}",
                    }
                ],
            )
            self.summary = resp.choices[0].message.content.strip()[:200]
            self.save()
        except Exception as e:
            logger.warning(f"Compression failed: {e}")

    def get_context(self) -> str:
        parts = []
        # 核心状态
        if self.core["research_goal"]:
            parts.append(f"【调研目标】{self.core['research_goal']}")
        if self.core["key_findings"]:
            parts.append(
                "【关键发现】"
                + "; ".join(f["content"][:100] for f in self.core["key_findings"][-5:])
            )
        if self.summary:
            parts.append(f"【历史摘要】{self.summary}")
        # 最近对话（按优先级排列）
        recent_sorted = sorted(
            self.recent, key=lambda m: PRIORITY.get(m.get("role", ""), 1), reverse=True
        )
        for m in recent_sorted[-self.max_recent * 2 :]:
            role = {"user": "用户", "assistant": "AI", "tool_result": "工具"}.get(
                m["role"], m["role"]
            )
            parts.append(f"[{role}]: {m['content'][:300]}")
        return "\n".join(parts)

    def clear(self):
        self.recent, self.summary = [], ""
        self.core = dict(CORE_TEMPLATE)
        self._path().unlink(missing_ok=True)


memories: dict[str, ConversationMemory] = {}


def get_memory(topic_id: str) -> ConversationMemory:
    if topic_id not in memories:
        memories[topic_id] = ConversationMemory(topic_id)
    return memories[topic_id]
