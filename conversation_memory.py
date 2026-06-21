"""多轮对话记忆管理：短期上下文 + 会话摘要 + 长期用户画像。"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("data/memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class ConversationMemory:
    """管理单个课题的多轮对话记忆：最近 N 轮完整 + 历史摘要。"""

    def __init__(self, topic_id: str, max_recent: int = 5):
        self.topic_id = topic_id
        self.max_recent = max_recent
        self.recent: list[dict[str, str]] = []  # [{"role":"user","content":"..."}]
        self.summary: str = ""  # 历史对话摘要
        self._load()

    def _path(self):
        return MEMORY_DIR / f"{self.topic_id}.json"

    def _load(self):
        """从磁盘恢复。"""
        p = self._path()
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self.recent = data.get("recent", [])
                self.summary = data.get("summary", "")
            except Exception:
                pass

    def save(self):
        self._path().write_text(
            json.dumps(
                {
                    "recent": self.recent,
                    "summary": self.summary,
                },
                ensure_ascii=False,
            )
        )

    def add_turn(self, question: str, answer: str):
        """添加一轮对话。"""
        self.recent.append({"role": "user", "content": question})
        self.recent.append({"role": "assistant", "content": answer})

        # 超过 max_recent 轮 → 生成摘要
        while len(self.recent) > self.max_recent * 2:
            old = self.recent.pop(0)
            self.recent.pop(0)
            if self.summary:
                self.summary += f" | 用户曾问: {old.get('content','')[:80]}"
            else:
                self.summary = f"历史对话: 用户曾问: {old.get('content','')[:80]}"

        self.save()

    def get_context(self) -> str:
        """获取当前对话上下文（用于 Prompt 拼接）。"""
        parts = []
        if self.summary:
            parts.append(f"【对话历史摘要】{self.summary}")
        for m in self.recent[-self.max_recent * 2 :]:
            role = "用户" if m["role"] == "user" else "AI"
            parts.append(f"[{role}]: {m['content'][:300]}")
        return "\n".join(parts)

    def clear(self):
        self.recent = []
        self.summary = ""
        self._path().unlink(missing_ok=True)


# 全局记忆仓库
memories: dict[str, ConversationMemory] = {}


def get_memory(topic_id: str) -> ConversationMemory:
    if topic_id not in memories:
        memories[topic_id] = ConversationMemory(topic_id)
    return memories[topic_id]
